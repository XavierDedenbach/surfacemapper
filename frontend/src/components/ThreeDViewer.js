import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

/**
 * ThreeDViewer component for 3D surface visualization using Three.js
 * Renders surfaces stacked vertically according to their Z-coordinates
 */

// Utility to validate UTM coordinates
export function validateUTMCoordinates(vertices) {
  if (!vertices || vertices.length === 0) return false;
  const xs = vertices.map(v => v[0]);
  const ys = vertices.map(v => v[1]);
  return xs.every(x => Math.abs(x) > 180) && ys.every(y => Math.abs(y) > 90);
}

const ThreeDViewer = ({ 
  analysisResult,
  onPointHover, 
  onMouseLeave,
  onBack,
  selectedSurfaces
}) => {
  const surfaces = analysisResult?.surfaces || [];
  const analysisBoundary = analysisResult?.analysis_boundary || null;

  // Validate props
  const data = {
    surfaces: Array.isArray(surfaces) ? surfaces : [],
    analysisBoundary: analysisBoundary || null,
    selectedSurfaces: Array.isArray(selectedSurfaces) ? selectedSurfaces : []
  };

  const callbacks = {
    onPointHover: onPointHover || (() => {}),
    onMouseLeave: onMouseLeave || (() => {})
  };

  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());

  const [isInitialized, setIsInitialized] = useState(false);
  const [visibleSurfaces, setVisibleSurfaces] = useState(new Set(data.surfaces.map((_, i) => i)));

  // Handle point selection with validation
  const handlePointSelect = useCallback((point) => {
    if (point && callbacks.onPointHover) {
      callbacks.onPointHover(point.x, point.y, point.z);
    }
  }, [callbacks]);

  // Handle surface visibility toggle with validation
  const handleSurfaceToggle = useCallback((surfaceIndex) => {
    const newVisibleSurfaces = new Set(visibleSurfaces);
    if (newVisibleSurfaces.has(surfaceIndex)) {
      newVisibleSurfaces.delete(surfaceIndex);
    } else {
      newVisibleSurfaces.add(surfaceIndex);
    }
    setVisibleSurfaces(newVisibleSurfaces);
  }, [visibleSurfaces]);

  // Initialize Three.js scene with validation
  useEffect(() => {
    if (!mountRef.current) return;
    const mountNode = mountRef.current;

    try {
      // Scene setup
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x1a1a1a);
      sceneRef.current = scene;

      // Camera setup
      const camera = new THREE.PerspectiveCamera(
        75,
        mountNode.clientWidth / mountNode.clientHeight,
        0.1,
        2000 // Increased far clipping plane
      );
      camera.position.set(15, 15, 15);
      cameraRef.current = camera;

      // Renderer setup
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(mountNode.clientWidth, mountNode.clientHeight);
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

      mountNode.appendChild(renderer.domElement);
      setIsInitialized(true);

      // Animation loop
      const animate = () => {
        requestAnimationFrame(animate);
        if (controlsRef.current) {
          controlsRef.current.update();
        }
        if (rendererRef.current && sceneRef.current && cameraRef.current) {
          rendererRef.current.render(sceneRef.current, cameraRef.current);
        }
      };
      animate();

      // Cleanup
      return () => {
        if (mountNode && renderer.domElement) {
          mountNode.removeChild(renderer.domElement);
        }
        if (renderer) {
          renderer.dispose();
        }
      };
    } catch (error) {
      console.error('Error initializing Three.js scene:', error);
    }
  }, []);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (!mountRef.current || !rendererRef.current || !cameraRef.current) return;

      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;

      const camera = cameraRef.current;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();

      rendererRef.current.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Coordinate system validation (only in browser, not in test/SSR)
  useEffect(() => {
    if (typeof window === 'undefined' || typeof document === 'undefined') return;
    if (!surfaces || surfaces.length === 0) return;
    surfaces.forEach((surface, index) => {
      if (surface.vertices && surface.vertices.length > 0) {
        const isUTM = validateUTMCoordinates(surface.vertices);
        if (!isUTM) {
          console.warn(`WGS84 coordinates detected in surface ${index}. Visualization expects UTM (meters).`);
        }
      }
    });
  }, [surfaces]);

  // Render surfaces
  useEffect(() => {
    if (!isInitialized || !surfaces || surfaces.length === 0) return;

    const scene = sceneRef.current;
    
    // Create a group for the surface meshes
    const surfaceGroup = new THREE.Group();

    // Clear existing surface meshes
    scene.children = scene.children.filter(child => 
      !(child instanceof THREE.Group) && // Keep everything that is not our surface group
      child.type !== 'Mesh'
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
      mesh.visible = visibleSurfaces.has(index);
      surfaceGroup.add(mesh);
    });

    scene.add(surfaceGroup);

    // Update camera position to fit all surfaces in the group
    if (surfaceGroup.children.length > 0) {
      const box = new THREE.Box3().setFromObject(surfaceGroup);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = 75;
      const cameraZ = Math.abs(maxDim / 2 / Math.tan((fov * Math.PI) / 360));

      const camera = cameraRef.current;
      if (camera) {
        camera.position.set(center.x, center.y + size.y * 0.2, center.z + cameraZ * 1.5);
        camera.lookAt(center);
        if (controlsRef.current) {
          controlsRef.current.target.copy(center);
          controlsRef.current.update();
        }
      }
    }
  }, [surfaces, isInitialized, visibleSurfaces]);

  // Handle mouse interaction for point analysis
  useEffect(() => {
    if (!isInitialized || !mountRef.current) return;

    console.log('ThreeDViewer: useEffect for mouse events running. isInitialized:', isInitialized, 'mountRef:', !!mountRef.current);
    const canvas = mountRef.current.querySelector('canvas');
    if (canvas) {
      console.log('ThreeDViewer: Canvas found, attaching mousemove and mouseleave listeners');
      canvas.addEventListener('mousemove', handleMouseMove);
      canvas.addEventListener('mouseleave', handleMouseLeave);
      return () => {
        canvas.removeEventListener('mousemove', handleMouseMove);
        canvas.removeEventListener('mouseleave', handleMouseLeave);
      };
    } else {
      console.log('ThreeDViewer: Canvas NOT found, cannot attach listeners');
    }
  }, [isInitialized, handlePointSelect, callbacks]);

  // Log when isInitialized is set to true
  useEffect(() => {
    if (isInitialized) {
      console.log('ThreeDViewer: isInitialized set to true');
    }
  }, [isInitialized]);

  // Helper to recursively collect all meshes in the scene
  function getAllMeshes(object) {
    let meshes = [];
    if (object.type === 'Mesh') {
      meshes.push(object);
    }
    if (object.children && object.children.length > 0) {
      object.children.forEach(child => {
        meshes = meshes.concat(getAllMeshes(child));
      });
    }
    return meshes;
  }

    const handleMouseMove = (event) => {
    console.log('ThreeDViewer: handleMouseMove event fired');
      const rect = mountRef.current.getBoundingClientRect();
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      // Raycasting for point analysis
      if (cameraRef.current) {
        raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
      } else {
        return; // Camera not ready
      }

    // Use helper to get all meshes in the scene
    const meshes = getAllMeshes(sceneRef.current);
    const intersects = raycasterRef.current.intersectObjects(meshes);

      if (intersects.length > 0) {
        const point = intersects[0].point;
      // Debug log for intersection
      console.log('Raycast intersection found:', point.x, point.y, point.z, event.clientX - rect.left, event.clientY - rect.top);
      // Provide both 3D and 2D mouse position (relative to container)
      if (callbacks.onPointHover) {
        console.log('Calling onPointHover callback');
        callbacks.onPointHover(point.x, point.y, point.z, event.clientX - rect.left, event.clientY - rect.top);
      } else {
        console.log('onPointHover callback not available');
      }
    } else {
      console.log('No raycast intersection found');
    }
  };

  const handleMouseLeave = () => {
    if (typeof callbacks.onMouseLeave === 'function') {
      callbacks.onMouseLeave();
    }
  };

  return (
    <div className="h-full w-full relative">
      <div ref={mountRef} className="h-full w-full" />
      <div className="absolute top-2 left-2 bg-gray-800 bg-opacity-75 p-2 rounded">
        <h4 className="text-white text-sm font-bold mb-2">Surfaces</h4>
        {data.surfaces.map((surface, index) => (
          <div key={index} className="flex items-center text-white">
            <input
              type="checkbox"
              id={`surface-${index}`}
              checked={visibleSurfaces.has(index)}
              onChange={() => handleSurfaceToggle(index)}
              className="mr-2"
            />
            <label htmlFor={`surface-${index}`}>{surface.name || `Surface ${index + 1}`}</label>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ThreeDViewer; 