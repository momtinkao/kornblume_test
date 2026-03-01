<script setup lang="ts">
import { ref, onMounted, shallowRef } from 'vue';
import * as ort from 'onnxruntime-web';
import Tesseract, { createWorker } from 'tesseract.js';
import { useWarehouseStore } from '@/stores/warehouseStore'; 

const fileInput = ref<HTMLInputElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);

const isImporting = ref(false);
const loadingText = ref('');
const detectedItems = ref<{ name: string; quantity: number | string; conf: number }[]>([]);

// CV2 輪廓篩選參數 (與 detect_v4.py 一致)
const MIN_SIZE = 300;
const MAX_SIZE = 500;
const ASPECT_RATIO = 0.25;

// 保存模型與設定
const onnxSession = shallowRef<ort.InferenceSession | null>(null);
const id2label = ref<Record<string, string>>({});
const ocrWorker = shallowRef<Tesseract.Worker | null>(null);

const warehouseStore = useWarehouseStore();

// 初始化模型
onMounted(async () => {
  loadingText.value = '正在載入 ONNX 模型與標籤...';
  try {
    // 1. 載入標籤
    const configRes = await fetch('/models/config.json');
    const configData = await configRes.json();
    id2label.value = configData.id2label || {};

    // 2. 載入 ONNX 模型 (設定 WASM 路徑避免 404)
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    onnxSession.value = await ort.InferenceSession.create('/models/model.onnx', {
      executionProviders: ['wasm']
    });

    // 3. 載入 Tesseract OCR (限定數字)
    loadingText.value = '正在載入 OCR 引擎...';
    const worker = await createWorker('eng');
    await worker.setParameters({ tessedit_char_whitelist: '0123456789' });
    ocrWorker.value = worker;

    loadingText.value = '';
  } catch (error) {
    console.error('初始化失敗:', error);
    loadingText.value = '載入模型失敗，請確認檔案已放置於 public/models/';
  }
});

const triggerFileInput = () => {
  if (fileInput.value) fileInput.value.click();
};

// 將 ImageData 轉為 ONNX 需要的 Tensor (取代 Numpy 正規化)
const preprocessImageToTensor = (imageData: ImageData) => {
  const { data, width, height } = imageData;
  const float32Data = new Float32Array(3 * width * height);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    float32Data[i] = (r - mean[0]) / std[0]; // R
    float32Data[width * height + i] = (g - mean[1]) / std[1]; // G
    float32Data[2 * width * height + i] = (b - mean[2]) / std[2]; // B
  }
  return new ort.Tensor('float32', float32Data, [1, 3, height, width]);
};

// 主要處理邏輯
const processImage = async (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  
  // 忽略下一行的 any 檢查，因為透過 CDN 引入的 OpenCV 本來就沒有 TypeScript 型別
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cv = (window as any).cv;
  
  if (!file || !onnxSession.value || !ocrWorker.value || !cv) {
      if(!cv) alert("OpenCV 尚未載入完成，請稍後再試！");
      return;
  }

  isImporting.value = true;
  detectedItems.value = [];
  loadingText.value = '分析圖片中...';

  const img = new Image();
  img.src = URL.createObjectURL(file);
  await new Promise((resolve) => (img.onload = resolve));

  // --- CV2 尋找輪廓 ---
  const src = cv.imread(img);
  if (canvasRef.value) cv.imshow(canvasRef.value, src);

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

  // 排序：先按 Y 排序，再按 X 排序 (取代 Python 的 sort_contours)
  validBoxes.sort((a, b) => {
      if (Math.abs(a.y - b.y) > a.height * 0.5) return a.y - b.y;
      return a.x - b.x;
  });

  gray.delete(); blurred.delete(); edges.delete(); kernel.delete(); contours.delete(); hierarchy.delete();

  loadingText.value = `找到 ${validBoxes.length} 個素材，正在辨識...`;

  const cropCanvas = document.createElement('canvas');
  const cropCtx = cropCanvas.getContext('2d')!;

  for (let i = 0; i < validBoxes.length; i++) {
    const rect = validBoxes[i];
    const p = 2; // crop padding
    
    // 1. 裁切區域
    cropCanvas.width = rect.width - 2 * p;
    cropCanvas.height = rect.height - 2 * p;
    cropCtx.drawImage(img, rect.x + p, rect.y + p, cropCanvas.width, cropCanvas.height, 0, 0, cropCanvas.width, cropCanvas.height);
    
    // 2. ONNX 圖片預處理 (縮放到 224x224)
    const onnxCanvas = document.createElement('canvas');
    onnxCanvas.width = 224; onnxCanvas.height = 224;
    const onnxCtx = onnxCanvas.getContext('2d')!;
    onnxCtx.drawImage(cropCanvas, 0, 0, 224, 224);
    
    const tensor = preprocessImageToTensor(onnxCtx.getImageData(0, 0, 224, 224));

    // 3. 執行推論
    const results = await onnxSession.value.run({ [onnxSession.value.inputNames[0]]: tensor });
    const logits = results[onnxSession.value.outputNames[0]].data as Float32Array;
    
    // 找尋最高分與信心度 (Softmax)
    let maxLogit = -Infinity;
    for (const val of logits) if (val > maxLogit) maxLogit = val;
    let sumExp = 0;
    const probs = logits.map(v => {
      const exp = Math.exp(v - maxLogit);
      sumExp += exp;
      return exp;
    }).map(v => v / sumExp);
    
    const topClassId = probs.indexOf(Math.max(...probs));
    const conf = probs[topClassId];
    const matName = id2label.value[topClassId.toString()] || `Unknown_${topClassId}`;

    if (conf < 0.7) continue;

    // 4. OCR 數量辨識
    const ocrCanvas = document.createElement('canvas');
    const roiH = Math.floor(cropCanvas.height * 0.4);
    ocrCanvas.width = cropCanvas.width; ocrCanvas.height = roiH;
    const ocrCtx = ocrCanvas.getContext('2d')!;
    // 對應 Python 的 roi = img_crop[h - roi_h : h, 0:w] 並放大3倍
    ocrCtx.drawImage(cropCanvas, 0, cropCanvas.height - roiH, cropCanvas.width, roiH, 0, 0, cropCanvas.width, roiH);
    
    const { data: { text } } = await ocrWorker.value.recognize(ocrCanvas.toDataURL());
    const qty = parseInt(text.trim());

    detectedItems.value.push({
      name: matName,
      quantity: isNaN(qty) ? '未偵測到' : qty,
      conf
    });
  }

  isImporting.value = false;
  loadingText.value = '辨識完成！';
  src.delete(); 
};

// 寫入倉庫的邏輯
const saveToWarehouse = () => {
    let updateCount = 0;
    detectedItems.value.forEach(item => {
        if (typeof item.quantity === 'number') {
            // 利用現有的 store 函數更新庫存
            warehouseStore.updateItem(item.name, item.quantity);
            updateCount++;
        }
    });
    alert(`已成功更新 ${updateCount} 項物品至倉庫！`);
};
</script>

<template>
  <div class="custom-border custom-gradient-gray-blue p-4 rounded-lg flex flex-col items-center">
    <div class="flex flex-wrap space-x-2 sm:space-x-3 gap-y-3 items-center justify-center">
      <input type="file" ref="fileInput" @change="processImage" accept="image/*" class="hidden" />
      
      <button 
        @click="triggerFileInput" 
        :disabled="isImporting || !onnxSession"
        class="bg-gradient-to-br from-success to-green-600 focus:ring-2 hover:bg-gradient-to-bl text-white font-bold py-2 px-4 rounded disabled:opacity-50">
        <i class="fa-solid fa-camera mr-2"></i> 上傳截圖辨識
      </button>

      <button 
        v-if="detectedItems.length > 0"
        @click="saveToWarehouse"
        class="bg-gradient-to-br from-info to-blue-600 focus:ring-2 hover:bg-gradient-to-bl text-white font-bold py-2 px-4 rounded">
        <i class="fa-solid fa-floppy-disk mr-2"></i> 儲存至倉庫
      </button>
    </div>

    <!-- 狀態提示 -->
    <div v-if="loadingText" class="text-white mt-4 font-bold flex items-center">
      <img v-if="isImporting" src="/images/items/common/vertin-wheel.apng" alt="Loading" class="w-8 h-8 mr-2"/>
      {{ loadingText }}
    </div>

    <!-- 預覽圖 Canvas (隱藏或縮小顯示) -->
    <div class="w-full max-w-lg mt-4 hidden sm:block overflow-hidden rounded-lg">
      <canvas ref="canvasRef" class="w-full h-auto"></canvas>
    </div>

    <!-- 辨識結果表格 -->
    <div v-if="detectedItems.length > 0" class="w-full mt-4 overflow-x-auto hidden-scrollbar">
      <table class="table-auto w-full text-white text-sm">
        <thead>
          <tr class="border-b border-white/20">
            <th class="px-2 py-2">物品名稱</th>
            <th class="px-2 py-2 text-info">辨識數量</th>
            <th class="px-2 py-2 text-gray-400">信心度</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(item, index) in detectedItems" :key="index" class="border-b border-white/10 hover:bg-white/5">
            <td class="px-2 py-2 text-center">{{ $t ? $t(item.name) : item.name }}</td>
            <td class="px-2 py-2 text-center text-info font-bold">{{ item.quantity }}</td>
            <td class="px-2 py-2 text-center text-gray-400">{{ (item.conf * 100).toFixed(1) }}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>