/**
 * SkinScan AI — Backend API Client
 *
 * Connects the React frontend to the FastAPI backend for real ML inference.
 * Manages SQLite JWT/Bearer token sessions, persistent scan history, and HITL feedback.
 * Falls back gracefully to client-side demo mode if backend is offline.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ─────────────────────────────────────────────────────────
// AUTH & SESSION STATE HELPERS
// ─────────────────────────────────────────────────────────

export function getToken() {
  return localStorage.getItem('skinscan_auth_token') || null;
}

export function setToken(token) {
  if (token) {
    localStorage.setItem('skinscan_auth_token', token);
  } else {
    localStorage.removeItem('skinscan_auth_token');
  }
}

export function getUser() {
  const raw = localStorage.getItem('skinscan_user');
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function setUser(user) {
  if (user) {
    localStorage.setItem('skinscan_user', JSON.stringify(user));
  } else {
    localStorage.removeItem('skinscan_user');
  }
}

export function clearAuth() {
  setToken(null);
  setUser(null);
}

// ─────────────────────────────────────────────────────────
// AUTHENTICATION API
// ─────────────────────────────────────────────────────────

export async function authRegister({ name, email, password, role = 'clinician' }) {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password, role }),
  });
  const data = await response.json();
  if (!response.ok) {
    const errorMsg = typeof data.detail === 'string' ? data.detail : 'Registration failed';
    throw new Error(errorMsg);
  }
  setToken(data.token);
  setUser(data.user);
  return data;
}

export async function authLogin({ email, password }) {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const data = await response.json();
  if (!response.ok) {
    const errorMsg = typeof data.detail === 'string' ? data.detail : 'Invalid email or password';
    throw new Error(errorMsg);
  }
  setToken(data.token);
  setUser(data.user);
  return data;
}

export async function authGetMe() {
  const token = getToken();
  if (!token) return null;
  try {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) {
      clearAuth();
      return null;
    }
    const data = await response.json();
    setUser(data.user);
    return data.user;
  } catch {
    return null;
  }
}

// ─────────────────────────────────────────────────────────
// SCAN HISTORY & PERSISTENCE
// ─────────────────────────────────────────────────────────

export async function getScans() {
  const token = getToken();
  if (!token) return [];
  try {
    const response = await fetch(`${API_BASE_URL}/scans`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return data.scans || [];
  } catch (err) {
    console.warn('[SkinScan API] Failed to fetch scan history:', err);
    return [];
  }
}

export async function deleteScan(scanId) {
  const token = getToken();
  if (!token) throw new Error('Authentication required to delete scans');
  const response = await fetch(`${API_BASE_URL}/scans/${scanId}`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `Failed to delete scan: HTTP ${response.status}`);
  }
  return await response.json();
}

// ─────────────────────────────────────────────────────────
// HITL BIOPSY FEEDBACK & MLOPS RETRAINING QUEUE
// ─────────────────────────────────────────────────────────

export async function submitBiopsyFeedback(scanId, { confirmedBiopsyLabel, clinicianNotes = '' }) {
  const token = getToken();
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const response = await fetch(`${API_BASE_URL}/scans/${scanId}/feedback`, {
    method: 'PATCH',
    headers,
    body: JSON.stringify({
      confirmed_biopsy_label: confirmedBiopsyLabel,
      clinician_notes: clinicianNotes,
    }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || 'Failed to submit biopsy feedback');
  }
  return data;
}

export async function getRetrainQueue() {
  try {
    const response = await fetch(`${API_BASE_URL}/mlops/retrain-queue`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (err) {
    console.warn('[SkinScan API] Failed to fetch retrain queue:', err);
    return { queued_count: 0, model_version: 'v4 (frozen)', cases: [] };
  }
}

export async function exportRetrainQueue() {
  const response = await fetch(`${API_BASE_URL}/mlops/export-queue`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return await response.json();
}

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
 * Automatically attaches Authorization header if user is signed in to persist scan.
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
    formData.append('doctor_preference', patientData.doctorPreference || 'auto');
    formData.append('lesion_type', patientData.lesionType || 'mole');

    // Attach Bearer token if logged in (for SQLite persistence)
    const headers = {};
    const token = getToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      headers,
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
    scanId: backendResult.scan_id || null,
    isPersisted: Boolean(backendResult.is_persisted),
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
    hairRemovedImage: backendResult.hair_removed_image || backendResult.images?.dullrazor,
    hairMaskImage: backendResult.hair_mask_image || backendResult.images?.hair_mask,
    gradcamImage: backendResult.gradcam_image || backendResult.images?.gradcam,
    verification: backendResult.verification,
    cancerVerification: backendResult.cancer_verification,
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
