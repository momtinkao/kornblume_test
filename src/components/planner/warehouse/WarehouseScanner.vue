<script setup lang="ts">
import { ref } from 'vue';
import { useWarehouseStore } from '@/stores/warehouseStore';

const fileInput = ref<HTMLInputElement | null>(null);

const isImporting = ref(false);
const loadingText = ref('');
const currentImageIndex = ref(0);
const totalImages = ref(0);

// 辨識結果清單
const detectedItems = ref<{ name: string; quantity: number | string; conf: number; ocrImage: string }[]>([]);

// CV2 輪廓篩選參數
const MIN_SIZE = 300;
const MAX_SIZE = 500;
const ASPECT_RATIO = 0.25;

const warehouseStore = useWarehouseStore();

const triggerFileInput = () => {
  if (fileInput.value) fileInput.value.click();
};

/**
 * 處理單張圖片 (只負責 OpenCV 找輪廓與裁切，AI 交給後端)
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

    // 明確宣告 any[] 避免 TypeScript 報錯
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const validBoxes: any[] = [];
    for (let i = 0; i < contours.size(); ++i) {
        const rect = cv.boundingRect(contours.get(i));
        const ratio = rect.width / rect.height;
        if (rect.width >= MIN_SIZE && rect.height >= MIN_SIZE && 
            rect.width <= MAX_SIZE && rect.height <= MAX_SIZE &&
            ratio >= (1 - ASPECT_RATIO) && ratio <= (1 + ASPECT_RATIO)) {
            validBoxes.push(rect);
        }
    }
    
    // 根據 Y 座標與 X 座標進行排序 (從上到下，從左到右)
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

        // ==========================================
        // 呼叫本機 Python FastAPI 伺服器進行 AI 推論
        // ==========================================
        try {
            const blob = await new Promise<Blob | null>(resolve => { cropCanvas.toBlob(resolve, 'image/png'); });
            if (blob) {
                const formData = new FormData();
                formData.append('file', blob);
                
                // 呼叫本機端 Python 伺服器 API
                const resp = await fetch('http://127.0.0.1:5000/ocr', { method: 'POST', body: formData });
                const data = await resp.json();
                
                if (data.error) {
                    matName = data.name || "伺服器錯誤";
                    conf = 1; 
                    qty = '後端錯誤';
                } else {
                    matName = data.name || 'Unknown';
                    conf = data.conf || 0;
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
            }
        } catch (e) {
            console.error("Local Server API Error", e);
            matName = "API 斷線";
            conf = 1; 
            qty = '連線失敗';
        }

        // 過濾信心度過低的雜訊
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

    if (Object.keys(aggregated).length === 0) {
        alert("沒有偵測到任何有效的物品數量，無法存入倉庫！\n(請確認 OCR 預覽中的數字是否正確)");
        return;
    }

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
            <td class="px-2 py-3 text-center font-medium font-mono text-gray-200" :class="{'text-red-400': item.quantity === '連線失敗' || item.quantity === '後端錯誤'}">
              <div class="text-base leading-tight">
                {{ $t(item.name) }} 
              </div>
              <div class="text-[10px] text-gray-500 mt-1 opacity-70">{{ item.name }}</div>
            </td>
            <td class="px-2 py-3 text-center text-info font-bold text-lg" :class="{'text-red-500 text-sm': typeof item.quantity === 'string'}">{{ item.quantity }}</td>
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