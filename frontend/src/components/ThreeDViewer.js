import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

/**
 * ThreeDViewer component for 3D surface visualization using Three.js
 * Renders surfaces stacked vertically according to their Z-coordinates
 */
const ThreeDViewer = ({ surfaces, analysisBoundary, onPointHover, selectedSurfaces }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());

  const [isInitialized, setIsInitialized] = useState(false);

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(10, 10, 10);

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    // Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(20, 20);
    scene.add(gridHelper);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    mountRef.current.appendChild(renderer.domElement);
    setIsInitialized(true);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (!mountRef.current || !rendererRef.current || !sceneRef.current) return;

      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;

      const camera = sceneRef.current.children.find(child => child.type === 'PerspectiveCamera');
      if (camera) {
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      }

      rendererRef.current.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Render surfaces
  useEffect(() => {
    if (!isInitialized || !surfaces || surfaces.length === 0) return;

    const scene = sceneRef.current;
    
    // Clear existing surface meshes
    scene.children = scene.children.filter(child => 
      child.type === 'AmbientLight' || 
      child.type === 'DirectionalLight' || 
      child.type === 'GridHelper' || 
      child.type === 'AxesHelper'
    );

    // Surface colors
    const colors = [
      0x4285f4, // Blue
      0xea4335, // Red
      0xfbbc04, // Yellow
      0x34a853  // Green
    ];

    // Create surface meshes
    surfaces.forEach((surface, index) => {
      if (!surface.vertices || surface.vertices.length === 0) return;

      // Create geometry from vertices
      const geometry = new THREE.BufferGeometry();
      const vertices = new Float32Array(surface.vertices.flat());
      geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

      // Create faces if available
      if (surface.faces && surface.faces.length > 0) {
        const indices = new Uint32Array(surface.faces.flat());
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
      } else {
        // Create simple triangulation for point clouds
        // TODO: Implement proper triangulation
        geometry.computeBoundingSphere();
      }

      // Create material
      const material = new THREE.MeshLambertMaterial({
        color: colors[index % colors.length],
        transparent: true,
        opacity: 0.8,
        side: THREE.DoubleSide
      });

      // Create mesh
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = { surfaceIndex: index, surfaceData: surface };

      scene.add(mesh);
    });

    // Update camera position to fit all surfaces
    const box = new THREE.Box3().setFromObject(scene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = 75;
    const cameraZ = Math.abs(maxDim / 2 / Math.tan((fov * Math.PI) / 360));

    const camera = scene.children.find(child => child.type === 'PerspectiveCamera');
    if (camera) {
      camera.position.set(center.x + cameraZ, center.y + cameraZ, center.z + cameraZ);
      camera.lookAt(center);
      controlsRef.current.target.copy(center);
      controlsRef.current.update();
    }
  }, [surfaces, isInitialized]);

  // Handle mouse interaction for point analysis
  useEffect(() => {
    if (!isInitialized || !mountRef.current) return;

    const handleMouseMove = (event) => {
      const rect = mountRef.current.getBoundingClientRect();
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      // Raycasting for point analysis
      raycasterRef.current.setFromCamera(mouseRef.current, 
        sceneRef.current.children.find(child => child.type === 'PerspectiveCamera')
      );

      const intersects = raycasterRef.current.intersectObjects(
        sceneRef.current.children.filter(child => child.type === 'Mesh')
      );

      if (intersects.length > 0) {
        const point = intersects[0].point;
        if (onPointHover) {
          onPointHover(point.x, point.y, point.z);
        }
      }
    };

    const canvas = mountRef.current.querySelector('canvas');
    if (canvas) {
      canvas.addEventListener('mousemove', handleMouseMove);
      return () => canvas.removeEventListener('mousemove', handleMouseMove);
    }
  }, [isInitialized, onPointHover]);

  return (
    <div className="three-d-viewer">
      <div 
        ref={mountRef} 
        style={{ 
          width: '100%', 
          height: '100%',
          minHeight: '400px'
        }} 
      />
      <div className="viewer-controls">
        <button onClick={() => {
          if (controlsRef.current) {
            controlsRef.current.reset();
          }
        }}>
          Reset View
        </button>
        <div className="surface-legend">
          {surfaces?.map((surface, index) => (
            <div key={index} className="legend-item">
              <div 
                className="legend-color" 
                style={{ 
                  backgroundColor: `#${Math.floor(surface.color || 0x4285f4).toString(16).padStart(6, '0')}` 
                }}
              />
              <span>Surface {index + 1}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ThreeDViewer; 