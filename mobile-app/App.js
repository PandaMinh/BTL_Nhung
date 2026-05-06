import { CameraView, useCameraPermissions } from "expo-camera";
import * as FileSystem from "expo-file-system/legacy";
import * as ImageManipulator from "expo-image-manipulator";
import { StatusBar } from "expo-status-bar";
import { useState } from "react";
import { ActivityIndicator, Alert, Pressable, StyleSheet, Text, View } from "react-native";

const BACKEND_URL = "http://192.168.1.11:8000";
const FRAME_COUNT = 8;
const SELECT_COUNT = 1;
const BURST_DURATION_MS = 3500;
const FRAME_INTERVAL_MS = Math.floor(BURST_DURATION_MS / FRAME_COUNT);
const REQUEST_TIMEOUT_MS = 30000;
const RETRY_TIMEOUT_MS = 45000;

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraRef, setCameraRef] = useState(null);
  const [facing, setFacing] = useState("back");
  const [submitting, setSubmitting] = useState(false);
  const [statusText, setStatusText] = useState("San sang");
  const [result, setResult] = useState(null);
  const [showResultScreen, setShowResultScreen] = useState(false);

  if (!permission) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#ffffff" />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.centered}>
        <StatusBar style="light" />
        <Pressable style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.buttonText}>Cho phep camera</Text>
        </Pressable>
      </View>
    );
  }

  async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(timeoutId);
    }
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function prepareFrame(uri) {
    const manipulated = await ImageManipulator.manipulateAsync(
      uri,
      [{ resize: { width: 480 } }],
      { compress: 0.45, format: ImageManipulator.SaveFormat.JPEG }
    );
    const info = await FileSystem.getInfoAsync(manipulated.uri, { size: true });
    return {
      uri: manipulated.uri,
      qualityScore: Number(info.size || 0),
      sizeBytes: Number(info.size || 0),
    };
  }

  async function checkBackendReachable() {
    const response = await fetchWithTimeout(`${BACKEND_URL}/health`, {}, 5000);
    if (!response.ok) {
      throw new Error("Backend khong phan hoi /health.");
    }
  }

  async function sendFrameWithRetry(frame, index) {
    const formData = new FormData();
    formData.append("reference_height_cm", "30");
    formData.append("image", {
      uri: frame.uri,
      name: `selected_${index + 1}.jpg`,
      type: "image/jpeg",
    });

    try {
      return await fetchWithTimeout(
        `${BACKEND_URL}/measure-height`,
        {
          method: "POST",
          body: formData,
        },
        REQUEST_TIMEOUT_MS
      );
    } catch (error) {
      if (error?.name !== "AbortError") {
        throw error;
      }
      // Retry once with longer timeout to avoid failing the whole burst.
      return await fetchWithTimeout(
        `${BACKEND_URL}/measure-height`,
        {
          method: "POST",
          body: formData,
        },
        RETRY_TIMEOUT_MS
      );
    }
  }

  async function handleCapture() {
    if (!cameraRef || submitting) return;

    try {
      setSubmitting(true);
      setStatusText("Dang chup burst local...");
      const captured = [];

      for (let i = 0; i < FRAME_COUNT; i += 1) {
        const photo = await cameraRef.takePictureAsync({
          quality: 0.7,
          skipProcessing: true,
          exif: false,
        });
        captured.push(photo.uri);
        if (i < FRAME_COUNT - 1) await sleep(FRAME_INTERVAL_MS);
      }

      setShowResultScreen(true);
      setResult({
        phase: "processing",
        message: "Da chup xong. Dang chon frame tot nhat tren iPhone...",
      });

      setResult({
        phase: "processing",
        message: "Dang kiem tra ket noi backend...",
      });
      await checkBackendReachable();

      const prepared = [];
      for (const uri of captured) {
        prepared.push(await prepareFrame(uri));
      }
      prepared.sort((a, b) => b.qualityScore - a.qualityScore);
      const topFrames = prepared.slice(0, SELECT_COUNT);

      setResult({
        phase: "processing",
        message: `Da chon ${topFrames.length} frame tot nhat. Dang gui backend...`,
        selected_frames: topFrames.map((f) => ({ sizeBytes: f.sizeBytes })),
      });

      const bestFrame = topFrames[0];
      setResult({
        phase: "processing",
        message: "Dang gui 1 frame tot nhat len backend...",
        selected_frames: [{ sizeBytes: bestFrame.sizeBytes }],
      });
      const response = await sendFrameWithRetry(bestFrame, 0);
      const payload = await response.json();
      if (!response.ok || payload?.height_final_cm === undefined || payload?.height_final_cm === null) {
        throw new Error("Khong co frame nao do duoc chieu cao.");
      }

      const finalPayload = {
        ...payload,
        height_final_cm: Number(payload.height_final_cm),
        burst_local: {
          captured_frames: FRAME_COUNT,
          sent_frames: 1,
          valid_frames: 1,
        },
        selected_frames: [{ sizeBytes: bestFrame.sizeBytes }],
        message: "Da gui 1 frame tot nhat va nhan ket qua.",
      };

      setResult(finalPayload);
      setStatusText(`${finalPayload.height_final_cm} cm`);
    } catch (error) {
      const message =
        error?.name === "AbortError"
          ? "Timeout backend. Da retry 1 lan nhung van that bai."
          : error.message || "Co loi mang.";
      setResult({ phase: "error", message });
      setStatusText(message);
      Alert.alert("Loi", message);
    } finally {
      setSubmitting(false);
    }
  }

  function buildExplanation(payload) {
    if (!payload || payload.phase === "processing" || payload.phase === "error") {
      return payload?.message || "Dang xu ly...";
    }
    const quality = payload.quality || {};
    const matQuality = Number(quality.mat_quality ?? 0);
    const poseVisibility = Number(quality.pose_visibility ?? 0);
    const bodyTilt = Number(quality.body_tilt_degrees ?? 0);
    const local = payload.burst_local || {};

    const lines = [];
    lines.push(
      `Da chup ${local.captured_frames ?? "--"} frame local, gui ${local.sent_frames ?? "--"} frame tot nhat.`
    );
    lines.push(
      `Ket qua cuoi dung median tren ${local.valid_frames ?? "--"} frame hop le: ${payload.height_final_cm ?? "--"} cm.`
    );
    lines.push(`Mat quality: ${matQuality.toFixed(3)}. Pose visibility: ${poseVisibility.toFixed(3)}.`);
    lines.push(`Do nghieng co the: ${bodyTilt.toFixed(2)} deg.`);
    return lines.join(" ");
  }

  if (showResultScreen) {
    const processing = result?.phase === "processing";
    const hasError = result?.phase === "error";
    return (
      <View style={styles.resultScreen}>
        <StatusBar style="light" />
        <View style={styles.resultCard}>
          <Text style={styles.resultTitle}>Chi tiet do</Text>
          <Text style={styles.resultHeight}>
            {processing || hasError ? "--" : `${result?.height_final_cm ?? "--"} cm`}
          </Text>
          <Text style={styles.resultMeta}>Thong bao: {result?.message || "Dang xu ly..."}</Text>
          <Text style={styles.resultMeta}>
            Local burst: {result?.burst_local?.captured_frames ?? "--"} frame
          </Text>
          <Text style={styles.resultMeta}>
            Gui backend: {result?.burst_local?.sent_frames ?? "--"} frame
          </Text>
          <Text style={styles.resultMeta}>
            Hop le: {result?.burst_local?.valid_frames ?? "--"} frame
          </Text>
          <Text style={styles.explainText}>{buildExplanation(result)}</Text>
          {processing ? <ActivityIndicator size="small" color="#93c5fd" /> : null}

          <Pressable
            style={[styles.actionButton, styles.primaryButton]}
            onPress={() => {
              setShowResultScreen(false);
              setResult(null);
            }}
          >
            <Text style={styles.buttonText}>Do lai</Text>
          </Pressable>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <StatusBar style="light" />
      <CameraView style={styles.camera} facing={facing} ref={setCameraRef} />

      <View style={styles.topStatus}>
        <Text style={styles.statusText}>{statusText}</Text>
      </View>

      <View style={styles.bottomBar}>
        <Pressable
          style={[styles.actionButton, styles.secondaryButton]}
          onPress={() => setFacing((current) => (current === "back" ? "front" : "back"))}
          disabled={submitting}
        >
          <Text style={styles.buttonText}>Doi camera</Text>
        </Pressable>

        <Pressable
          style={[styles.actionButton, styles.primaryButton, submitting && styles.disabledButton]}
          onPress={handleCapture}
          disabled={submitting}
        >
          <Text style={styles.buttonText}>{submitting ? "Dang chup..." : "Bat dau do"}</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#000000",
  },
  camera: {
    ...StyleSheet.absoluteFillObject,
  },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#000000",
  },
  permissionButton: {
    minWidth: 180,
    minHeight: 52,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#b45309",
    paddingHorizontal: 18,
  },
  topStatus: {
    position: "absolute",
    top: 60,
    left: 16,
    right: 16,
    borderRadius: 12,
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  statusText: {
    color: "#ffffff",
    fontSize: 14,
    textAlign: "center",
    fontWeight: "600",
  },
  bottomBar: {
    position: "absolute",
    left: 16,
    right: 16,
    bottom: 40,
    flexDirection: "row",
    gap: 12,
  },
  actionButton: {
    flex: 1,
    minHeight: 54,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 12,
  },
  primaryButton: {
    backgroundColor: "#b45309",
  },
  secondaryButton: {
    backgroundColor: "rgba(0,0,0,0.55)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.35)",
  },
  buttonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "700",
  },
  disabledButton: {
    opacity: 0.7,
  },
  resultScreen: {
    flex: 1,
    backgroundColor: "#0b1220",
    justifyContent: "center",
    padding: 16,
  },
  resultCard: {
    backgroundColor: "#111827",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#1f2937",
    padding: 16,
    gap: 8,
  },
  resultTitle: {
    color: "#93c5fd",
    fontSize: 18,
    fontWeight: "700",
  },
  resultHeight: {
    color: "#ffffff",
    fontSize: 34,
    fontWeight: "800",
  },
  resultMeta: {
    color: "#d1d5db",
    fontSize: 14,
  },
  explainText: {
    color: "#e5e7eb",
    fontSize: 14,
    lineHeight: 20,
    marginTop: 4,
    marginBottom: 6,
  },
});
