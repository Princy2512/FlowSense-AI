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
const PREDICT_SCRIPT = path.join(ROOT, "ml", "predict_server.py");
const CLIENT_BUILD_PATH = path.join(ROOT, "client", "dist");

console.log("=== Starting Server ===");
console.log("ROOT Directory:", ROOT);
console.log("ROOT contents:", fs.readdirSync(ROOT));
console.log("METADATA_PATH exists:", fs.existsSync(METADATA_PATH));
console.log("CLIENT_BUILD_PATH exists:", fs.existsSync(CLIENT_BUILD_PATH));

// Start persistent Python process ONCE
let pythonProcess;
let requestQueue = [];
let isProcessing = false;

function startPythonProcess() {
    console.log("Starting persistent Python process...");
    pythonProcess = spawn("python3", [PREDICT_SCRIPT], {
        cwd: ROOT,
        stdio: ["pipe", "pipe", "pipe"],
    });

    pythonProcess.stderr.on("data", (data) => {
        console.error("Python stderr:", data.toString());
    });

    pythonProcess.stdout.on("data", (data) => {
        const lines = data.toString().trim().split("\n");
        lines.forEach(line => {
            if (line.trim()) {
                if (requestQueue.length > 0) {
                    const { resolve } = requestQueue.shift();
                    isProcessing = false;
                    try {
                        const result = JSON.parse(line.trim());
                        resolve(result);
                    } catch (e) {
                        resolve({ error: "Invalid response from Python" });
                    }
                    processNextInQueue();
                }
            }
        });
    });

    pythonProcess.on("close", (code) => {
        console.error(`Python process closed with code ${code}, restarting...`);
        setTimeout(startPythonProcess, 1000);
    });

    pythonProcess.on("error", (err) => {
        console.error("Failed to start Python process:", err);
    });
}

startPythonProcess();

function processNextInQueue() {
    if (isProcessing || requestQueue.length === 0) {
        return;
    }
    isProcessing = true;
    const { payload } = requestQueue[0];

    try {
        pythonProcess.stdin.write(JSON.stringify(payload) + "\n");
    } catch (e) {
        const { reject } = requestQueue.shift();
        isProcessing = false;
        reject(e);
        processNextInQueue();
    }
}

function runPythonPredict(payload) {
    return new Promise((resolve, reject) => {
        requestQueue.push({ payload, resolve, reject });
        processNextInQueue();
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

