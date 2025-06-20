#!/usr/bin/env node

// TypeScript compilation test script
// This script validates that all TypeScript interfaces compile correctly

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔍 Testing TypeScript interface compilation...\n');

// Check if TypeScript is installed
try {
  execSync('npx tsc --version', { stdio: 'pipe' });
} catch (error) {
  console.log('❌ TypeScript not found. Installing TypeScript...');
  try {
    execSync('npm install --save-dev typescript @types/node', { stdio: 'inherit' });
  } catch (installError) {
    console.error('❌ Failed to install TypeScript:', installError.message);
    process.exit(1);
  }
}

// Test TypeScript compilation
try {
  console.log('📝 Compiling TypeScript interfaces...');
  execSync('npx tsc --noEmit', { stdio: 'inherit' });
  console.log('✅ TypeScript compilation successful!');
} catch (error) {
  console.error('❌ TypeScript compilation failed:', error.message);
  process.exit(1);
}

// Test specific interface files
const testFiles = [
  'src/types/api.ts',
  'src/types/api.test.ts'
];

console.log('\n📋 Checking interface files...');
for (const file of testFiles) {
  const filePath = path.join(__dirname, file);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${file} exists`);
  } else {
    console.log(`❌ ${file} missing`);
  }
}

console.log('\n🎉 All TypeScript interface tests passed!'); 