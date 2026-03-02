<script setup lang="ts">
import { ref, shallowRef } from 'vue';
import { useWarehouseStore } from '@/stores/warehouseStore';

// 核心組件載入
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ort: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let initOcr: any = null; 
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loadImg: any = null;

const fileInput = ref<HTMLInputElement | null>(null);

const isImporting = ref(false);
const loadingText = ref('');
const currentImageIndex = ref(0);
const totalImages = ref(0);

// 辨識模式切換：true 使用本地 GPU 伺服器, false 使用網頁內 WebGL/WASM 運算
const useLocalServer = ref(false); 

const detectedItems = ref<{ name: string; quantity: number | string; conf: number; ocrImage: string }[]>([]);

// CV2 輪廓篩選參數
const MIN_SIZE = 300;
const MAX_SIZE = 500;
const ASPECT_RATIO = 0.25;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const matSession = shallowRef<any>(null);
const id2label = ref<Record<string, string>>({});
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ocrEngine = shallowRef<any>(null); 
const isModelsLoaded = ref(false);

const warehouseStore = useWarehouseStore();


/**
 * 初始化 AI 模型
 */
const initModels = async (): Promise<boolean> => {
  if (isModelsLoaded.value) return true;
  loadingText.value = '正在下載 AI 核心組件 (僅首次辨識需要)...';
  
  try {
    const ortModule = await import('onnxruntime-web');
    ort = ortModule.default || ortModule;
    if (!ort.Tensor && ortModule.Tensor) ort = ortModule;

    const esearchOcr = await import('esearch-ocr');
    initOcr = esearchOcr.init || (esearchOcr.default && esearchOcr.default.init);
    loadImg = esearchOcr.loadImg || (esearchOcr.default && esearchOcr.default.loadImg);

    const baseUrl = import.meta.env.BASE_URL;

    loadingText.value = '正在載入素材分類模型...';
    const configRes = await fetch(`${baseUrl}models/config.json`);
    const configData = await configRes.json();
    id2label.value = configData.id2label || {};

    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    
    // 開啟 WebGL 加速
    matSession.value = await ort.InferenceSession.create(`${baseUrl}models/model.onnx`, {
      executionProviders: ['webgl', 'wasm']
    });

    loadingText.value = '正在讀取 OCR 字典檔...';
    const dicRes = await fetch(`${baseUrl}models/dict.txt`);
    const recDic = await dicRes.text(); 

    loadingText.value = '正在啟動 eSearch-OCR 引擎...';
    ocrEngine.value = await initOcr({
      ort,
      det: { input: `${baseUrl}models/det.onnx` },
      rec: {
        input: `${baseUrl}models/rec.onnx`,
        decodeDic: recDic,
        optimize: { space: false }
      }
    });
    
    isModelsLoaded.value = true;
    loadingText.value = '';
    return true;
  } catch (error) {
    console.error('初始化失敗:', error);
    loadingText.value = '模型載入失敗，請確認檔案完整性。';
    return false;
  }
};

const triggerFileInput = () => {
  if (fileInput.value) fileInput.value.click();
};

const preprocessImageToTensor = (imageData: ImageData) => {
  const { data, width, height } = imageData;
  const float32Data = new Float32Array(3 * width * height);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  for (let i = 0; i < width * height; i++) {
    float32Data[i] = (data[i * 4] / 255.0 - mean[0]) / std[0]; 
    float32Data[width * height + i] = (data[i * 4 + 1] / 255.0 - mean[1]) / std[1]; 
    float32Data[2 * width * height + i] = (data[i * 4 + 2] / 255.0 - mean[2]) / std[2]; 
  }
  return new ort.Tensor('float32', float32Data, [1, 3, height, width]);
};

/**
 * 處理單張圖片
 */
const processSingleImage = async (file: File) => {
    // @ts-expect-error: OpenCV 全域載入
    const cv = window.cv;
    if (!cv) return;

    loadingText.value = `[圖片 ${currentImageIndex.value}/${totalImages.value}] 分析佈局中...`;
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await new Promise((resolve) => { img.onload = resolve; });

    const src = cv.imread(img);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    const blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    const edges = new cv.Mat();
    cv.Canny(blurred, edges, 50, 150, 3, false);
    const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.dilate(edges, edges, kernel, new cv.Point(-1, -1), 1);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const validBoxes = [];
    for (let i = 0; i < contours.size(); ++i) {
        const rect = cv.boundingRect(contours.get(i));
        const ratio = rect.width / rect.height;
        if (rect.width >= MIN_SIZE && rect.height >= MIN_SIZE && 
            rect.width <= MAX_SIZE && rect.height <= MAX_SIZE &&
            ratio >= (1 - ASPECT_RATIO) && ratio <= (1 + ASPECT_RATIO)) {
            validBoxes.push(rect);
        }
    }
    validBoxes.sort((a, b) => (Math.abs(a.y - b.y) > a.height * 0.5 ? a.y - b.y : a.x - b.x));
    gray.delete(); blurred.delete(); edges.delete(); kernel.delete(); contours.delete(); hierarchy.delete();

    const cropCanvas = document.createElement('canvas');
    const cropCtx = cropCanvas.getContext('2d')!;

    for (let i = 0; i < validBoxes.length; i++) {
        if (i % 3 === 0) await new Promise(resolve => setTimeout(resolve, 0)); 
        loadingText.value = `[圖 ${currentImageIndex.value}/${totalImages.value}] 辨識第 ${i + 1} / ${validBoxes.length} 素材...`;

        const rect = validBoxes[i];
        cropCanvas.width = rect.width - 4;
        cropCanvas.height = rect.height - 4;
        cropCtx.drawImage(img, rect.x + 2, rect.y + 2, cropCanvas.width, cropCanvas.height, 0, 0, cropCanvas.width, cropCanvas.height);
        
        const ocrImageUrl = cropCanvas.toDataURL();

        let matName = '';
        let conf = 0;
        let qty: string | number = '未偵測到';

        if (useLocalServer.value) {
            // ==========================================
            // 方式二：本機 GPU 伺服器模式 (分類 + OCR 皆由後端處理)
            // ==========================================
            try {
                const blob = await new Promise<Blob | null>(resolve => { cropCanvas.toBlob(resolve, 'image/png'); });
                if (blob) {
                    const formData = new FormData();
                    formData.append('file', blob);
                    const resp = await fetch('http://127.0.0.1:5000/ocr', { method: 'POST', body: formData });
                    const data = await resp.json();
                    
                    // 解析回傳的分類資訊
                    matName = data.name || '';
                    conf = data.conf || 0;

                    // 確保將提取出來的字串轉換為數字
                    if (data.text) {
                        const cleanStr = data.text.replace(/[^0-9]/g, '');
                        if (cleanStr) {
                            const parsedNum = parseInt(cleanStr, 10);
                            // 防呆，確保不是 NaN 且合理 (<1000)
                            if (!isNaN(parsedNum) && parsedNum <= 1000) {
                                qty = parsedNum;
                            }
                        }
                    }
                }
            } catch (e) {
                console.error("Local Server API Error", e);
            }
        } else {
            // ==========================================
            // 方式一：網頁模式 (WebGL/WASM 本地運算)
            // ==========================================
            const matCanvas = document.createElement('canvas');
            matCanvas.width = 224; matCanvas.height = 224;
            const matCtx = matCanvas.getContext('2d')!;
            matCtx.drawImage(cropCanvas, 0, 0, 224, 224);
            const matTensor = preprocessImageToTensor(matCtx.getImageData(0, 0, 224, 224));
            const results = await matSession.value!.run({ [matSession.value!.inputNames[0]]: matTensor });
            const logits = results[matSession.value!.outputNames[0]].data as Float32Array;
            
            let maxLogit = -Infinity;
            for (const val of logits) if (val > maxLogit) maxLogit = val;
            let sumExp = 0;
            const probs = logits.map((v: number) => {
                const exp = Math.exp(v - maxLogit);
                sumExp += exp;
                return exp;
            }).map((v: number) => v / sumExp);
            
            const topClassId = probs.indexOf(Math.max(...probs));
            conf = probs[topClassId];
            matName = (id2label.value[topClassId.toString()] || `Unknown_${topClassId}`).trim();

            if (conf >= 0.7) {
                const ocrRoiCanvas = document.createElement('canvas');
                const baseRoiH = Math.floor(cropCanvas.height * 0.35);
                const baseRoiY = cropCanvas.height - baseRoiH;
                ocrRoiCanvas.width = cropCanvas.width; ocrRoiCanvas.height = baseRoiH;
                const ocrRoiCtx = ocrRoiCanvas.getContext('2d')!;
                ocrRoiCtx.drawImage(cropCanvas, 0, baseRoiY, cropCanvas.width, baseRoiH, 0, 0, ocrRoiCanvas.width, ocrRoiCanvas.height);

                const roiMat = cv.imread(ocrRoiCanvas);
                const grayMat = new cv.Mat();
                cv.cvtColor(roiMat, grayMat, cv.COLOR_RGBA2GRAY);
                const scaledMat = new cv.Mat();
                cv.resize(grayMat, scaledMat, new cv.Size(0, 0), 2.5, 2.5, cv.INTER_CUBIC);
                const paddedMat = new cv.Mat();
                cv.copyMakeBorder(scaledMat, paddedMat, 30, 30, 30, 30, cv.BORDER_REPLICATE);

                const finalCanvas = document.createElement('canvas');
                cv.imshow(finalCanvas, paddedMat);
                
                roiMat.delete(); grayMat.delete(); scaledMat.delete(); paddedMat.delete();

                try {
                    const rawImg = await loadImg(finalCanvas);
                    const ocrResult = await ocrEngine.value!.ocr(rawImg);
                    if (ocrResult && ocrResult.parragraphs) {
                        const validNumbers: { num: number, cy: number }[] = [];
                        ocrResult.parragraphs.forEach((item: { text?: string, box?: [number, number][] }) => {
                            if (item && item.text) {
                                const cleanText = item.text.replace(/[^0-9]/g, '');
                                if (cleanText) {
                                    const num = parseInt(cleanText, 10);
                                    if (!isNaN(num) && num <= 1000) {
                                        const cy = Array.isArray(item.box) ? (item.box.reduce((sum, pt) => sum + pt[1], 0) / 4) : 0;
                                        validNumbers.push({ num, cy });
                                    }
                                }
                            }
                        });
                        if (validNumbers.length > 0) {
                            validNumbers.sort((a, b) => b.cy - a.cy);
                            qty = validNumbers[0].num;
                        }
                    }
                } catch (err) {
                    console.error("Web OCR Error:", err);
                }
            }
        }

        // 共用信心度過濾器
        if (conf < 0.8) continue;

        detectedItems.value.push({
            name: matName,
            quantity: qty,
            conf,
            ocrImage: ocrImageUrl 
        });
    }
    src.delete();
};

const processImage = async (event: Event) => {
  const files = (event.target as HTMLInputElement).files;
  if (!files || files.length === 0) return;

  isImporting.value = true;
  detectedItems.value = [];
  totalImages.value = files.length;
  
  // 如果不是伺服器模式，確保模型有載入
  if (!useLocalServer.value && !isModelsLoaded.value) {
      const success = await initModels();
      if (!success) { isImporting.value = false; return; }
  }

  for (let i = 0; i < files.length; i++) {
      currentImageIndex.value = i + 1;
      await processSingleImage(files[i]);
  }

  isImporting.value = false;
  loadingText.value = `辨識完成！共處理 ${totalImages.value} 張截圖。`;
  if (fileInput.value) fileInput.value.value = '';
};

const saveToWarehouse = () => {
    let updateCount = 0;
    const aggregated: Record<string, number> = {};
    detectedItems.value.forEach(item => {
        if (typeof item.quantity === 'number') {
            const storeItemId = item.name;
            aggregated[storeItemId] = (aggregated[storeItemId] || 0) + item.quantity;
        }
    });
    Object.entries(aggregated).forEach(([itemId, qty]) => {
        warehouseStore.updateItem(itemId, qty);
        updateCount++;
    });
    alert(`已將 ${updateCount} 項物品成功存入倉庫！`);
};

const clearResults = () => {
    detectedItems.value = [];
    if (fileInput.value) fileInput.value.value = ''; 
    loadingText.value = '';
};
</script>

<template>
  <div class="custom-border custom-gradient-gray-blue p-4 rounded-lg flex flex-col items-center w-full">
    <div class="flex flex-wrap space-x-2 sm:space-x-3 gap-y-3 items-center justify-center">
      <input type="file" ref="fileInput" @change="processImage" accept="image/*" multiple class="hidden" />
      <button @click="triggerFileInput" :disabled="isImporting" class="bg-gradient-to-br from-success to-green-600 focus:ring-2 hover:bg-gradient-to-bl text-white font-bold py-2 px-6 rounded shadow-lg disabled:opacity-50 transition-all duration-200 active:scale-95">
        <i class="fa-solid fa-camera mr-2"></i> 批次辨識
      </button>
      <button v-if="detectedItems.length > 0" @click="saveToWarehouse" class="bg-gradient-to-br from-info to-blue-600 focus:ring-2 hover:bg-gradient-to-bl text-white font-bold py-2 px-6 rounded shadow-lg transition-all duration-200 active:scale-95">
        <i class="fa-solid fa-floppy-disk mr-2"></i> 批次儲存
      </button>
      <button v-if="detectedItems.length > 0" @click="clearResults" class="bg-gradient-to-br from-red-600 to-red-800 focus:ring-2 hover:bg-gradient-to-bl text-white font-bold py-2 px-6 rounded shadow-lg transition-all duration-200 active:scale-95">
        <i class="fa-solid fa-trash mr-2"></i> 清除
      </button>
      <div class="flex items-center text-white text-xs ml-4 opacity-50 hover:opacity-100 cursor-pointer" @click="useLocalServer = !useLocalServer">
          <i :class="useLocalServer ? 'fa-solid fa-toggle-on text-success' : 'fa-solid fa-toggle-off'"></i>
          <span class="ml-1 font-bold">本地 GPU 加速模式</span>
      </div>
    </div>
    
    <div v-if="loadingText" class="text-white mt-4 font-bold flex items-center bg-black/30 px-4 py-2 rounded-full border border-white/10 transition-opacity duration-300">
      <img v-if="isImporting" src="/images/items/common/vertin-wheel.apng" alt="Loading" class="w-6 h-6 mr-3"/>
      <span class="text-sm tracking-wide">{{ loadingText }}</span>
    </div>

    <div v-if="detectedItems.length > 0" class="w-full mt-4 max-h-96 overflow-y-auto overflow-x-auto border border-white/20 rounded-md shadow-inner custom-scrollbar">
      <table class="table-auto w-full text-white text-sm relative border-collapse">
        <thead class="sticky top-0 bg-gray-900/95 z-10 shadow-sm">
          <tr class="border-b border-white/20">
            <th class="px-2 py-3">名稱 (ID)</th>
            <th class="px-2 py-3 text-info">數量</th>
            <th class="px-2 py-3 text-warning">原圖預覽</th>
            <th class="px-2 py-3 text-gray-400">信心度</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-white/10">
          <tr v-for="(item, index) in detectedItems" :key="index" class="bg-[#121b31]/50 hover:bg-white/10 transition-colors duration-150">
            <td class="px-2 py-3 text-center font-medium font-mono text-gray-200">{{$t(item.name) }}</td>
            <td class="px-2 py-3 text-center text-info font-bold text-lg">{{ item.quantity }}</td>
            <td class="px-2 py-3 flex justify-center items-center">
              <img :src="item.ocrImage" class="inline-block h-20 border border-white/30 rounded bg-white/10 object-contain shadow-sm" alt="ROI" />
            </td>
            <td class="px-2 py-3 text-center text-gray-400 font-mono">{{ (item.conf * 100).toFixed(0) }}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>