/**
 * SkinScan AI — Backend API Client
 *
 * Connects the React frontend to the FastAPI backend for real ML inference.
 * Falls back gracefully to client-side demo mode if backend is offline.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ─────────────────────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────────────────────

/**
 * Check if the backend is online and has a model loaded.
 * @returns {{ online: boolean, modelLoaded: boolean, device: string, gpuName: string|null }}
 */
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return {
      online: true,
      modelLoaded: data.model_loaded || false,
      device: data.device || 'unknown',
      gpuName: data.gpu_name || null,
      lastInferenceMs: data.last_inference_ms || 0,
      weightsExist: data.weights_exist || false,
    };
  } catch {
    return {
      online: false,
      modelLoaded: false,
      device: 'offline',
      gpuName: null,
      lastInferenceMs: 0,
      weightsExist: false,
    };
  }
}

// ─────────────────────────────────────────────────────────
// MAIN ANALYSIS ENDPOINT
// ─────────────────────────────────────────────────────────

/**
 * Send an image + patient metadata to the backend for real ML inference.
 *
 * @param {File|Blob} imageFile - The skin lesion image file
 * @param {Object} patientData - Patient metadata
 * @param {number} patientData.age
 * @param {string} patientData.sex
 * @param {string} patientData.localization
 * @param {number} patientData.durationMonths
 * @param {boolean} patientData.historyOfMelanoma
 * @returns {Promise<{success: boolean, result: Object|null, error: string|null, isRealInference: boolean}>}
 */
export async function analyzeWithBackend(imageFile, patientData = {}) {
  try {
    const formData = new FormData();

    // Attach image file
    if (imageFile instanceof File || imageFile instanceof Blob) {
      formData.append('file', imageFile, 'scan.jpg');
    } else if (typeof imageFile === 'string' && imageFile.startsWith('data:')) {
      // Convert base64 data URL to Blob
      const blob = await dataUrlToBlob(imageFile);
      formData.append('file', blob, 'scan.jpg');
    } else {
      return {
        success: false,
        result: null,
        error: 'Invalid image input',
        isRealInference: false,
      };
    }

    // Attach patient metadata as form fields
    formData.append('age', String(patientData.age || 50));
    formData.append('sex', patientData.sex || 'male');
    formData.append('localization', patientData.localization || 'back');
    formData.append('duration_months', String(patientData.durationMonths || 6));
    formData.append('history_of_melanoma', String(patientData.historyOfMelanoma || false));

    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(30000), // 30s timeout for GPU inference
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const detail =
        typeof errorData.detail === 'string'
          ? errorData.detail
          : errorData.detail?.message || `Server error: ${response.status}`;
      return {
        success: false,
        result: null,
        error: detail,
        isRealInference: false,
      };
    }

    const result = await response.json();

    // Map backend snake_case response to frontend camelCase format
    return {
      success: true,
      result: mapBackendResultToFrontend(result),
      error: null,
      isRealInference: true,
    };
  } catch (err) {
    console.warn('[SkinScan API] Backend call failed:', err.message);
    return {
      success: false,
      result: null,
      error:
        err.name === 'TimeoutError'
          ? 'Backend inference timed out (30s). Is the GPU processing?'
          : `Backend offline: ${err.message}`,
      isRealInference: false,
    };
  }
}

// ─────────────────────────────────────────────────────────
// RESPONSE MAPPER
// ─────────────────────────────────────────────────────────

/**
 * Map the backend's snake_case JSON response to the frontend's
 * camelCase state shape that all UI components consume.
 */
function mapBackendResultToFrontend(backendResult) {
  return {
    id: backendResult.id,
    name: backendResult.name,
    shortName: backendResult.shortName || backendResult.name,
    code: backendResult.code,
    confidence: backendResult.confidence,
    probabilities: backendResult.probabilities,
    riskStage: backendResult.riskStage,
    riskStageNum: backendResult.riskStageNum,
    riskLevel: backendResult.riskLevel,
    spectralScore: backendResult.spectralScore,
    urgencyLevel: backendResult.urgencyLevel,
    urgencyColor: backendResult.urgencyColor,
    actionNeeded: backendResult.actionNeeded,
    dermasensorAssessment: backendResult.dermasensorAssessment,
    icd10: backendResult.icd10,
    category: backendResult.category,
    abcd: backendResult.abcd,
    referral: backendResult.referral,
    clinicalNotes: backendResult.clinicalNotes,
    metadata: backendResult.metadata
      ? {
          age: backendResult.metadata.age,
          sex: backendResult.metadata.sex,
          localization: backendResult.metadata.localization,
          durationMonths: backendResult.metadata.duration_months,
          historyOfMelanoma: backendResult.metadata.history_of_melanoma,
        }
      : undefined,
    images: backendResult.images || {},
    analyzedAt: backendResult.analyzedAt,
    inferenceTimeMs: backendResult.inference_time_ms,
    modelVersion: backendResult.modelVersion,
    imagePattern: backendResult.imagePattern,
    isRealInference: true,
  };
}

// ─────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────

/**
 * Convert a data URL (base64 image) to a Blob.
 */
async function dataUrlToBlob(dataUrl) {
  const response = await fetch(dataUrl);
  return response.blob();
}

/**
 * Get the current backend connection status as a display string.
 */
export function getBackendStatusLabel(health) {
  if (!health.online) return '⚠️ Backend Offline — Running in Demo Mode';
  if (!health.modelLoaded) return '⚠️ Model Not Loaded — Place weights in models/ directory';
  return `✅ Live AI (${health.device}${health.gpuName ? ` • ${health.gpuName}` : ''})`;
}
