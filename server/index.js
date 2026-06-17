import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import express from "express";
import cors from "cors";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const ROOT = path.resolve(__dirname, "..");
const METADATA_PATH = path.join(ROOT, "ml", "artifacts", "metadata.json");
const MODEL_PATH = path.join(ROOT, "ml", "artifacts", "model.joblib");
const PREDICT_SCRIPT = path.join(ROOT, "ml", "predict.py");
const CLIENT_BUILD_PATH = path.join(ROOT, "client", "dist");

console.log("=== Starting Server ===");
console.log("ROOT Directory:", ROOT);
console.log("ROOT contents:", fs.readdirSync(ROOT));
console.log("METADATA_PATH exists:", fs.existsSync(METADATA_PATH));
console.log("MODEL_PATH exists:", fs.existsSync(MODEL_PATH));
console.log("CLIENT_BUILD_PATH exists:", fs.existsSync(CLIENT_BUILD_PATH));
if (fs.existsSync(CLIENT_BUILD_PATH)) {
  console.log("CLIENT_BUILD_PATH contents:", fs.readdirSync(CLIENT_BUILD_PATH));
}
if (fs.existsSync(path.join(ROOT, "ml"))) {
  console.log("ML directory contents:", fs.readdirSync(path.join(ROOT, "ml")));
}
if (fs.existsSync(path.join(ROOT, "ml", "artifacts"))) {
  console.log("Artifacts directory contents:", fs.readdirSync(path.join(ROOT, "ml", "artifacts")));
}

function resolvePythonBin() {
  console.log("Resolving Python binary...");
  // Just use python3 directly since that's what Render has
  console.log("Using python3");
  return "python3";
}

function runPythonPredict(payload) {
  return new Promise((resolve, reject) => {
    const pythonBin = resolvePythonBin();
    console.log("Spawning Python with:", pythonBin, PREDICT_SCRIPT);
    console.log("Payload:", JSON.stringify(payload));

    const py = spawn(pythonBin, [PREDICT_SCRIPT], {
      cwd: ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let out = "";
    let err = "";
    py.stdout.on("data", (d) => {
      out += d.toString();
      console.log("PYTHON STDOUT:", d.toString());
    });
    py.stderr.on("data", (d) => {
      err += d.toString();
      console.error("PYTHON STDERR:", d.toString());
    });

    py.on("error", (e) => {
      console.error("PYTHON ERROR:", e);
      reject(e);
    });
    py.on("close", (code) => {
      console.log("PYTHON exited with code:", code);
      console.log("PYTHON stdout (full):", out);
      console.log("PYTHON stderr (full):", err);

      if (code !== 0) {
        reject(new Error(err || `Python exited with code ${code}`));
        return;
      }

      try {
        const trimmedOutput = out.trim();
        if (!trimmedOutput) {
          reject(new Error("No output from Python"));
          return;
        }
        const parsed = JSON.parse(trimmedOutput);
        resolve(parsed);
      } catch (e) {
        reject(new Error(`Failed to parse Python output: ${e.message}\nOutput: ${out}\nStderr: ${err}`));
      }
    });

    try {
      py.stdin.write(JSON.stringify(payload));
      py.stdin.end();
    } catch (e) {
      reject(new Error(`Failed to write payload: ${e.message}`));
    }
  });
}

app.get("/api/health", (_req, res) => {
  console.log("Received health check");
  res.json({ ok: true });
});

app.get("/api/metadata", (_req, res) => {
  console.log("Received metadata request");
  if (!fs.existsSync(METADATA_PATH)) {
    console.log("Metadata NOT found at:", METADATA_PATH);
    res.status(404).json({
      error: "metadata_not_found",
      message: "Model metadata not found. Please ensure training completed.",
    });
    return;
  }
  console.log("Serving metadata");
  const raw = fs.readFileSync(METADATA_PATH, "utf-8");
  res.type("json").send(raw);
});

app.post("/api/predict", async (req, res) => {
  console.log("Received prediction request:", req.body);
  try {
    const result = await runPythonPredict(req.body ?? {});
    if (result?.error) {
      res.status(400).json(result);
      return;
    }
    console.log("Prediction result:", result);
    res.json(result);
  } catch (e) {
    console.error("Prediction failed:", e);
    res.status(500).json({ 
      error: "predict_failed", 
      message: String(e?.message ?? e),
      details: e.stack
    });
  }
});

// Serve static files from the client build directory
if (fs.existsSync(CLIENT_BUILD_PATH)) {
  console.log("Serving static files from:", CLIENT_BUILD_PATH);
  app.use(express.static(CLIENT_BUILD_PATH));

  // For SPA routing, send index.html for any non-API routes
  app.get("*", (_req, res) => {
    res.sendFile(path.join(CLIENT_BUILD_PATH, "index.html"));
  });
} else {
  console.log("Client build path not found!");
  app.get("*", (_req, res) => {
    res.send("Client build not found. Please run the build process.");
  });
}

const PORT = process.env.PORT ? Number(process.env.PORT) : 5175;
app.listen(PORT, () => {
  console.log(`✅ Server successfully running on http://localhost:${PORT}`);
});

