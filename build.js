import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;
const CLIENT_DIR = path.join(ROOT, 'client');
const ML_DIR = path.join(ROOT, 'ml');

console.log('=== Starting Build ===');
console.log('ROOT:', ROOT);
console.log('Current directory contents:', fs.readdirSync(ROOT));

// Step 1: Install npm dependencies
console.log('\n--- Installing npm dependencies ---');
execSync('npm install', { cwd: ROOT, stdio: 'inherit' });
execSync('npm install', { cwd: CLIENT_DIR, stdio: 'inherit' });

// Step 2: Install Python dependencies and train model
console.log('\n--- Installing Python dependencies ---');
try {
  execSync('python3 --version', { stdio: 'inherit' });
  execSync('python3 -m pip install --upgrade pip', { cwd: ROOT, stdio: 'inherit' });
  execSync('python3 -m pip install -r ml/requirements.txt', { cwd: ROOT, stdio: 'inherit' });
  
  console.log('\n--- Training ML model ---');
  execSync('python3 ml/train.py', { cwd: ROOT, stdio: 'inherit' });
} catch (e) {
  console.error('Python step failed:', e.message);
}

// Step 3: Build client
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

const artifactsPath = path.join(ML_DIR, 'artifacts');
console.log('Artifacts path:', artifactsPath);
console.log('Artifacts exists:', fs.existsSync(artifactsPath));
if (fs.existsSync(artifactsPath)) {
  console.log('Artifacts contents:', fs.readdirSync(artifactsPath));
}

console.log('\n=== Build Complete ===');
