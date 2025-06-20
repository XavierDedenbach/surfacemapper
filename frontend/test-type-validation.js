#!/usr/bin/env node

// TypeScript type validation test script
// This script validates that TypeScript interfaces enforce proper types

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔍 Testing TypeScript type validation...\n');

// Create a temporary test file with invalid types to ensure type checking works
const tempTestFile = path.join(__dirname, 'temp-type-test.ts');

const invalidTypeTests = `
// This file should fail TypeScript compilation due to invalid types
import { SurfaceUploadResponse, GeoreferenceParams } from './src/types/api';

// Test 1: Invalid status enum
const invalidStatus: SurfaceUploadResponse = {
  message: "test",
  filename: "test.ply",
  status: "invalid_status" // This should cause a type error
};

// Test 2: Invalid number type
const invalidGeoref: GeoreferenceParams = {
  wgs84_lat: "not a number", // This should cause a type error
  wgs84_lon: -74.0060,
  orientation_degrees: 45.0,
  scaling_factor: 1.5
};

// Test 3: Missing required field
const incompleteResponse: SurfaceUploadResponse = {
  message: "test",
  // filename is missing - should cause a type error
  status: "completed"
};
`;

try {
  // Write the invalid test file
  fs.writeFileSync(tempTestFile, invalidTypeTests);
  
  console.log('📝 Testing type validation with invalid types...');
  
  // This should fail compilation
  execSync(`npx tsc --noEmit ${tempTestFile}`, { stdio: 'pipe' });
  
  console.log('❌ Type validation failed - invalid types should have caused compilation errors');
  process.exit(1);
  
} catch (error) {
  // Expected to fail due to invalid types
  console.log('✅ Type validation working correctly - invalid types properly rejected');
  
  // Clean up temp file
  if (fs.existsSync(tempTestFile)) {
    fs.unlinkSync(tempTestFile);
  }
}

// Test valid types
console.log('\n📝 Testing valid type compilation...');
try {
  execSync('npx tsc --noEmit src/types/api.test.ts', { stdio: 'inherit' });
  console.log('✅ Valid types compile successfully');
} catch (error) {
  console.error('❌ Valid types failed to compile:', error.message);
  process.exit(1);
}

console.log('\n🎉 All TypeScript type validation tests passed!'); 