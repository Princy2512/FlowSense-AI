import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;
const CLIENT_DIR = path.join(ROOT, 'client');
const SERVER_DIR = path.join(ROOT, 'server');
const ML_DIR = path.join(ROOT, 'ml');

console.log('=== Starting Build ===');
console.log('ROOT:', ROOT);
console.log('Current directory contents:', fs.readdirSync(ROOT));

// Step 1: Install npm dependencies
console.log('\n--- Installing npm dependencies ---');
execSync('npm install', { cwd: ROOT, stdio: 'inherit' });
execSync('npm install', { cwd: CLIENT_DIR, stdio: 'inherit' });

// Step 2: Install Python dependencies
console.log('\n--- Installing Python dependencies ---');
try {
  execSync('python3 --version', { stdio: 'inherit' });
  execSync('python3 -m pip install --upgrade pip', { cwd: ROOT, stdio: 'inherit' });
  execSync('python3 -m pip install -r ml/requirements.txt --user', { cwd: ROOT, stdio: 'inherit' });
} catch (e) {
  console.log('Python install warning (might be okay):', e.message);
}

// Step 3: Train ML model or use existing artifacts
console.log('\n--- Checking ML Artifacts ---');
const artifactsPath = path.join(ML_DIR, 'artifacts');
if (fs.existsSync(artifactsPath) && fs.readdirSync(artifactsPath).length > 0) {
  console.log('Artifacts already exist! Skipping training.');
} else {
  console.log('Training ML model...');
  try {
    execSync('python3 ml/train.py', { cwd: ROOT, stdio: 'inherit' });
  } catch (e) {
    console.error('ML training failed but continuing:', e.message);
  }
}

// Step 4: Build client
console.log('\n--- Building Client ---');
execSync('npm run build', { cwd: CLIENT_DIR, stdio: 'inherit' });

// Verify build output
console.log('\n--- Verifying Build ---');
const clientDistPath = path.join(CLIENT_DIR, 'dist');
console.log('Client dist path:', clientDistPath);
console.log('Client dist exists:', fs.existsSync(clientDistPath));
if (fs.existsSync(clientDistPath)) {
  console.log('Client dist contents:', fs.readdirSync(clientDistPath));
}

console.log('Artifacts path:', artifactsPath);
console.log('Artifacts exists:', fs.existsSync(artifactsPath));
if (fs.existsSync(artifactsPath)) {
  console.log('Artifacts contents:', fs.readdirSync(artifactsPath));
}

console.log('\n=== Build Complete ===');
