import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { gsap } from 'gsap';
import type { EmbeddingItem, LayoutAlgorithm } from './types';

let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let renderer: THREE.WebGLRenderer;
let controls: OrbitControls;
let loadedEmbeddings: EmbeddingItem[] = [];
let imageSprites: THREE.Sprite[] = [];
// const textureLoader = new THREE.TextureLoader(); // Not needed if using LoadingManager for all
let currentLayoutAlgorithm: LayoutAlgorithm = 'tsne';

const clusterEpicenters: Map<string, THREE.Vector3> = new Map();
const itemsByCluster: Map<string, EmbeddingItem[]> = new Map();

let pcaMin: THREE.Vector3 | null = null;
let pcaMax: THREE.Vector3 | null = null;
let pcaRange: THREE.Vector3 | null = null;
const DESIRED_PCA_VISUAL_SPREAD = 100; 

const SPRITE_SCALE = 5; 
const CLUSTER_EPICENTER_SPREAD = 50; 
const ITEMS_WITHIN_CLUSTER_SPREAD = 10; 

// This function will set up the basic Three.js environment
function initThreeApp() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff); 

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 50;

    const canvas = document.getElementById('webgl-canvas') as HTMLCanvasElement;
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputColorSpace = THREE.SRGBColorSpace; 

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = false; 
    controls.autoRotateSpeed = 0.5;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    window.addEventListener('resize', onWindowResize);
}

async function loadDataAndSetupUI() {
    try {
        const response = await fetch('/embeddings.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        loadedEmbeddings = await response.json();
        console.log(`Loaded ${loadedEmbeddings.length} embeddings.`);

        calculatePCABounds(); 
        createImageSprites(); // This starts the loading process via LoadingManager

        // const layoutSelect = document.getElementById('layout-algorithm') as HTMLSelectElement; // Removed as element no longer exists
        // if (layoutSelect) { // Added a null check for safety, though the block is removed
        //     layoutSelect.addEventListener('change', (event) => {
        //         currentLayoutAlgorithm = (event.target as HTMLSelectElement).value as LayoutAlgorithm;
        //         console.log(`Layout changed to: ${currentLayoutAlgorithm}`);
        //         if (['kmeans', 'dbscan', 'agglom'].includes(currentLayoutAlgorithm)) {
        //             let needsRecalculation = true;
        //             if (clusterEpicenters.size > 0) {
        //                 const firstKey = clusterEpicenters.keys().next().value;
        //                 if (firstKey && firstKey.startsWith(currentLayoutAlgorithm)) {
        //                     needsRecalculation = false;
        //                 }
        //             }
        //             if (needsRecalculation) {
        //                 calculateAndStoreClusterEpicenters(currentLayoutAlgorithm as 'kmeans' | 'dbscan' | 'agglom');
        //             }
        //         }
        //         updateSpritePositions();
        //     });
        // }

        const autoRotateCheckbox = document.getElementById('auto-rotate-toggle') as HTMLInputElement;
        autoRotateCheckbox.addEventListener('change', (event) => {
            if(controls) controls.autoRotate = (event.target as HTMLInputElement).checked;
        });

    } catch (error) {
        console.error("Failed to load or process embeddings.json:", error);
        const modalText = document.querySelector('#preloader-modal h1');
        if (modalText) {
            modalText.textContent = `Error loading data. Check console.`;
            modalText.parentElement?.querySelector('.loading-bar-container')?.remove();
        }
    }
}

function createImageSprites() {
    imageSprites.forEach(sprite => scene.remove(sprite)); // Clear existing sprites if any
    imageSprites = [];

    if (loadedEmbeddings.length === 0) {
        // Handle case where embeddings might be empty after filtering or error
        const startButton = document.getElementById('start-button');
        const loadingBarContainer = document.querySelector('.loading-bar-container');
        if (loadingBarContainer) (loadingBarContainer as HTMLElement).style.display = 'none';
        if (startButton) startButton.style.display = 'inline'; // Allow starting even if no images
        return;
    }

    const loadingManager = new THREE.LoadingManager();
    const localTextureLoader = new THREE.TextureLoader(loadingManager);

    loadingManager.onProgress = (url, itemsLoaded, itemsTotal) => {
        const progress = (itemsLoaded / itemsTotal) * 100;
        const loadingBarFill = document.getElementById('loading-bar-fill');
        if (loadingBarFill) {
            loadingBarFill.style.width = `${progress}%`;
        }
    };

    loadingManager.onLoad = () => {
        console.log("All textures loaded (or attempted).");
        const loadingBarContainer = document.querySelector('.loading-bar-container');
        if (loadingBarContainer) (loadingBarContainer as HTMLElement).style.display = 'none';
        
        if (['kmeans', 'dbscan', 'agglom'].includes(currentLayoutAlgorithm)) {
            calculateAndStoreClusterEpicenters(currentLayoutAlgorithm as 'kmeans' | 'dbscan' | 'agglom');
        }
        updateSpritePositions();

        if (renderer && scene && camera) {
            renderer.render(scene, camera); // Pre-render the scene
        }
        
        const startButton = document.getElementById('start-button');
        if (startButton) startButton.style.display = 'inline';
    };

    loadingManager.onError = (url) => { 
        console.error(`There was an error loading texture: ${url}`);
    };
    
    const cloudinaryBaseUrl = 'https://res.cloudinary.com/dazckbnuv/image/upload/latent/';

    loadedEmbeddings.forEach((item, index) => {
        // Assuming item.filename is like "no_prompt_0.webp"
        // And Cloudinary Public ID is "latent/no_prompt_0"
        // So, we want to construct "https://res.cloudinary.com/dazckbnuv/image/upload/latent/no_prompt_0.webp"
        const filenameWithoutExtension = item.filename.substring(0, item.filename.lastIndexOf('.'));
        const imageUrl = `${cloudinaryBaseUrl}${filenameWithoutExtension}.webp`;

        localTextureLoader.load(
            imageUrl,
            (texture) => {
                texture.colorSpace = THREE.SRGBColorSpace; 
                const material = new THREE.SpriteMaterial({ map: texture, transparent: true, alphaTest: 0.1 });
                const sprite = new THREE.Sprite(material);
                const aspectRatio = texture.image.width / texture.image.height;
                sprite.scale.set(SPRITE_SCALE * aspectRatio, SPRITE_SCALE, 1);
                sprite.userData = { id: index, embeddingItem: item }; 
                imageSprites.push(sprite);
                scene.add(sprite);
            },
            undefined, 
            (error) => { 
                console.error(`Failed to load texture for ${item.filename}:`, error);
            }
        );
    });
}

function calculatePCABounds() {
    if (loadedEmbeddings.length === 0) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const item of loadedEmbeddings) {
        minX = Math.min(minX, item.pca[0]); maxX = Math.max(maxX, item.pca[0]);
        minY = Math.min(minY, item.pca[1]); maxY = Math.max(maxY, item.pca[1]);
        minZ = Math.min(minZ, item.pca[2]); maxZ = Math.max(maxZ, item.pca[2]);
    }
    pcaMin = new THREE.Vector3(minX, minY, minZ);
    pcaMax = new THREE.Vector3(maxX, maxY, maxZ);
    let rangeX = maxX - minX; let rangeY = maxY - minY; let rangeZ = maxZ - minZ;
    if (rangeX === 0) rangeX = 1; if (rangeY === 0) rangeY = 1; if (rangeZ === 0) rangeZ = 1;
    pcaRange = new THREE.Vector3(rangeX, rangeY, rangeZ);
}

function calculateAndStoreClusterEpicenters(algorithm: 'kmeans' | 'dbscan' | 'agglom') {
    clusterEpicenters.clear(); itemsByCluster.clear();
    const uniqueClusterIds = new Set<number>();
    loadedEmbeddings.forEach(item => { uniqueClusterIds.add(item[algorithm as 'cluster_kmeans']); });
    const sortedClusterIds = Array.from(uniqueClusterIds).sort((a, b) => a - b);
    const numClusters = sortedClusterIds.length;
    const gridSize = Math.ceil(Math.sqrt(numClusters));
    sortedClusterIds.forEach((clusterId, index) => {
        const key = `${algorithm}_${clusterId}`;
        const x = (index % gridSize - (gridSize -1) / 2) * CLUSTER_EPICENTER_SPREAD;
        const y = (Math.floor(index / gridSize) - (gridSize-1) / 2) * CLUSTER_EPICENTER_SPREAD;
        clusterEpicenters.set(key, new THREE.Vector3(x, y, 0));
        const itemsInThisCluster = loadedEmbeddings.filter(item => item[algorithm as 'cluster_kmeans'] === clusterId);
        itemsByCluster.set(key, itemsInThisCluster);
    });
}

function getPositionInCluster(basePosition: THREE.Vector3, itemIndexInCluster: number, totalItemsInCluster: number): THREE.Vector3 {
    const n = totalItemsInCluster; const i = itemIndexInCluster;
    const phi = Math.acos(1 - 2 * (i + 0.5) / n); 
    const theta = Math.PI * (1 + Math.sqrt(5)) * (i + 0.5); 
    const x = ITEMS_WITHIN_CLUSTER_SPREAD * Math.sin(phi) * Math.cos(theta);
    const y = ITEMS_WITHIN_CLUSTER_SPREAD * Math.sin(phi) * Math.sin(theta);
    const z = ITEMS_WITHIN_CLUSTER_SPREAD * Math.cos(phi);
    return new THREE.Vector3(basePosition.x + x, basePosition.y + y, basePosition.z + z);
}

function updateSpritePositions() {
    if (imageSprites.length === 0 && loadedEmbeddings.length > 0 && !document.getElementById('start-button')?.style.display.includes('inline')) { 
        console.log("updateSpritePositions: Sprites not ready or loading not complete.");
        return; 
    }
    imageSprites.forEach((sprite) => {
        const item = sprite.userData.embeddingItem as EmbeddingItem;
        if (!item) { return; }
        let position: THREE.Vector3;
        switch (currentLayoutAlgorithm) {
            case 'tsne': position = new THREE.Vector3(...item.tsne); break;
            case 'pca':
                if (pcaMin && pcaMax && pcaRange) {
                    const normX = (item.pca[0] - pcaMin.x) / pcaRange.x; 
                    const normY = (item.pca[1] - pcaMin.y) / pcaRange.y; 
                    const normZ = (item.pca[2] - pcaMin.z) / pcaRange.z; 
                    position = new THREE.Vector3(
                        (normX - 0.5) * DESIRED_PCA_VISUAL_SPREAD, 
                        (normY - 0.5) * DESIRED_PCA_VISUAL_SPREAD,
                        (normZ - 0.5) * DESIRED_PCA_VISUAL_SPREAD
                    );
                } else { position = new THREE.Vector3(...item.pca); }
                break;
            case 'kmeans': case 'dbscan': case 'agglom':
                const clusterId = item[currentLayoutAlgorithm as 'cluster_kmeans']; 
                const clusterKey = `${currentLayoutAlgorithm}_${clusterId}`;
                const epicenter = clusterEpicenters.get(clusterKey);
                const itemsInThisSpecificCluster = itemsByCluster.get(clusterKey);
                if (epicenter && itemsInThisSpecificCluster) {
                    const itemIndexWithinCluster = itemsInThisSpecificCluster.findIndex(d => d.filename === item.filename);
                    if (itemIndexWithinCluster !== -1) {
                        position = getPositionInCluster(epicenter, itemIndexWithinCluster, itemsInThisSpecificCluster.length);
                    } else { position = new THREE.Vector3(0,0,0); }
                } else { position = new THREE.Vector3(0,0,0); }
                break;
            default: position = new THREE.Vector3(0,0,0);
        }
        sprite.position.copy(position);
    });
}

function onWindowResize() {
    if (camera && renderer) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

let animationFrameId: number | null = null;
function animate() {
    animationFrameId = requestAnimationFrame(animate);
    if (controls) controls.update(); 
    
    imageSprites.forEach(sprite => {
        if (sprite.material instanceof THREE.SpriteMaterial) {
            sprite.material.opacity = 1.0; 
        }
    });
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

function setupPreloader() {
    const modal = document.getElementById('preloader-modal');
    const startButton = document.getElementById('start-button');
    
    if (!modal || !startButton) {
        console.error("Preloader elements not found! Attempting to start app directly.");
        initThreeApp(); 
        loadDataAndSetupUI(); 
        animate();
        return;
    }
    
    initThreeApp(); 
    loadDataAndSetupUI(); 

    startButton.addEventListener('click', () => {
        gsap.to(modal, { 
            opacity: 0, 
            duration: 0.5, 
            onComplete: () => {
                modal.style.display = 'none';
                if (controls) { 
                    controls.target.set(0,0,0); 
                    controls.update();
                }
                if (animationFrameId === null) { 
                    animate(); 
                }
            }
        });
    });
}

// Start the application by setting up the preloader
setupPreloader();
