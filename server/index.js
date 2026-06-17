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
const PREDICT_SCRIPT = path.join(ROOT, "ml", "predict.py");
const VENV_PYTHON = path.join(ROOT, ".venv", "bin", "python");
const CLIENT_BUILD_PATH = path.join(ROOT, "client", "dist");

console.log("=== Starting Server ===");
console.log("ROOT:", ROOT);
console.log("METADATA_PATH exists:", fs.existsSync(METADATA_PATH));
console.log("CLIENT_BUILD_PATH exists:", fs.existsSync(CLIENT_BUILD_PATH));
if (fs.existsSync(CLIENT_BUILD_PATH)) {
  console.log("CLIENT_BUILD_PATH contents:", fs.readdirSync(CLIENT_BUILD_PATH));
}

function resolvePythonBin() {
  if (fs.existsSync(VENV_PYTHON)) {
    console.log("Using venv python:", VENV_PYTHON);
    return VENV_PYTHON;
  }
  console.log("Using system python3");
  return "python3";
}

function runPythonPredict(payload) {
  return new Promise((resolve, reject) => {
    const py = spawn(resolvePythonBin(), [PREDICT_SCRIPT], {
      cwd: ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let out = "";
    let err = "";
    py.stdout.on("data", (d) => {
      out += d.toString();
      console.log("Python stdout:", d.toString());
    });
    py.stderr.on("data", (d) => {
      err += d.toString();
      console.error("Python stderr:", d.toString());
    });

    py.on("error", (e) => {
      console.error("Python error:", e);
      reject(e);
    });
    py.on("close", (code) => {
      console.log("Python exited with code:", code);
      if (code !== 0 && !out.trim()) {
        reject(new Error(err || `python exited with code ${code}`));
        return;
      }
      try {
        const parsed = JSON.parse(out.trim());
        resolve(parsed);
      } catch (e) {
        reject(new Error(`Invalid python output: ${out}\n${err}`));
      }
    });

    py.stdin.write(JSON.stringify(payload));
    py.stdin.end();
  });
}

app.get("/api/health", (_req, res) => res.json({ ok: true }));

app.get("/api/metadata", (_req, res) => {
  if (!fs.existsSync(METADATA_PATH)) {
    res.status(404).json({
      error: "metadata_not_found",
      message: "Run: python ml/train.py (creates ml/artifacts/metadata.json)",
    });
    return;
  }
  const raw = fs.readFileSync(METADATA_PATH, "utf-8");
  res.type("json").send(raw);
});

app.post("/api/predict", async (req, res) => {
  try {
    const result = await runPythonPredict(req.body ?? {});
    if (result?.error) {
      res.status(400).json(result);
      return;
    }
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: "predict_failed", message: String(e?.message ?? e) });
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
  console.log(`Server running on http://localhost:${PORT}`);
});

