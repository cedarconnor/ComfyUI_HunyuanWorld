// HunyuanWorld 3D Viewer Utilities
// Utility functions for 3D viewer integration with ComfyUI

/**
 * HunyuanWorld data type constants
 */
export const DATA_TYPES = {
    PANORAMA_IMAGE: 'PANORAMA_IMAGE',
    SCENE_3D: 'SCENE_3D', 
    WORLD_MESH: 'WORLD_MESH',
    MODEL_HUNYUAN: 'MODEL_HUNYUAN',
    LAYERED_SCENE_3D: 'LAYERED_SCENE_3D',
    OBJECT_LABELS: 'OBJECT_LABELS',
    SCENE_MASK: 'SCENE_MASK',
    LAYER_MESH: 'LAYER_MESH'
};

/**
 * Viewer command constants
 */
export const VIEWER_COMMANDS = {
    SET_VIEW_MODE: 'set_view_mode',
    SET_RENDER_MODE: 'set_render_mode',
    SET_QUALITY: 'set_quality',
    RESET_VIEW: 'reset_view',
    EXPORT_VIEW: 'export_view',
    EXPORT_MESH: 'export_mesh',
    TOGGLE_LAYER: 'toggle_layer',
    SET_LAYER_OPACITY: 'set_layer_opacity',
    FOCUS_OBJECT: 'focus_object'
};

/**
 * View mode constants
 */
export const VIEW_MODES = {
    PANORAMA: 'panorama',
    WIREFRAME: 'wireframe',
    SOLID: 'solid',
    TEXTURED: 'textured',
    LAYERED: 'layered',
    DEPTH: 'depth'
};

/**
 * Render quality constants
 */
export const RENDER_QUALITY = {
    LOW: 'low',
    MEDIUM: 'medium',
    HIGH: 'high',
    ULTRA: 'ultra'
};

/**
 * Data conversion utilities
 */
export class DataConverter {
    /**
     * Convert tensor data to JavaScript arrays
     */
    static tensorToArray(tensor, shape = null) {
        if (!tensor) return null;
        
        // Handle different tensor formats
        if (Array.isArray(tensor)) {
            return tensor;
        }
        
        if (tensor.data && Array.isArray(tensor.data)) {
            return tensor.data;
        }
        
        // Handle buffer/typed arrays
        if (tensor.buffer || tensor instanceof ArrayBuffer) {
            const array = Array.from(new Float32Array(tensor));
            return shape ? this.reshapeArray(array, shape) : array;
        }
        
        return null;
    }
    
    /**
     * Reshape flat array to multidimensional array
     */
    static reshapeArray(flatArray, shape) {
        if (!shape || shape.length === 0) return flatArray;
        
        const totalElements = shape.reduce((a, b) => a * b, 1);
        if (flatArray.length !== totalElements) {
            console.warn('Array length does not match shape dimensions');
            return flatArray;
        }
        
        // For now, handle 2D and 3D shapes
        if (shape.length === 2) {
            const [rows, cols] = shape;
            const result = [];
            for (let i = 0; i < rows; i++) {
                result.push(flatArray.slice(i * cols, (i + 1) * cols));
            }
            return result;
        }
        
        if (shape.length === 3) {
            const [depth, rows, cols] = shape;
            const result = [];
            for (let d = 0; d < depth; d++) {
                const layer = [];
                for (let r = 0; r < rows; r++) {
                    const startIdx = d * rows * cols + r * cols;
                    layer.push(flatArray.slice(startIdx, startIdx + cols));
                }
                result.push(layer);
            }
            return result;
        }
        
        return flatArray;
    }
    
    /**
     * Convert image tensor to data URL
     */
    static tensorToImageURL(tensor, format = 'png') {
        if (!tensor) return null;
        
        try {
            // Convert tensor to ImageData format
            const imageData = this.tensorToImageData(tensor);
            if (!imageData) return null;
            
            // Create canvas and draw image data
            const canvas = document.createElement('canvas');
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            
            const ctx = canvas.getContext('2d');
            ctx.putImageData(imageData, 0, 0);
            
            return canvas.toDataURL(`image/${format}`);
            
        } catch (error) {
            console.error('Error converting tensor to image URL:', error);
            return null;
        }
    }
    
    /**
     * Convert tensor to ImageData
     */
    static tensorToImageData(tensor) {
        if (!tensor) return null;
        
        // Assume tensor is in format [H, W, C] or [C, H, W]
        let data, width, height, channels;
        
        if (Array.isArray(tensor) && tensor.length > 0) {
            // Handle [H, W, C] format
            if (Array.isArray(tensor[0]) && Array.isArray(tensor[0][0])) {
                height = tensor.length;
                width = tensor[0].length;
                channels = tensor[0][0].length;
                
                // Flatten to RGBA format
                data = new Uint8ClampedArray(width * height * 4);
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const srcIdx = y * width + x;
                        const dstIdx = srcIdx * 4;
                        
                        // Convert normalized values to 0-255 range
                        data[dstIdx] = Math.round(tensor[y][x][0] * 255);     // R
                        data[dstIdx + 1] = Math.round(tensor[y][x][1] * 255); // G
                        data[dstIdx + 2] = Math.round(tensor[y][x][2] * 255); // B
                        data[dstIdx + 3] = channels > 3 ? Math.round(tensor[y][x][3] * 255) : 255; // A
                    }
                }
            }
        }
        
        if (!data || !width || !height) {
            console.warn('Could not parse tensor format for image conversion');
            return null;
        }
        
        return new ImageData(data, width, height);
    }
}

/**
 * Viewer communication utilities
 */
export class ViewerCommunication {
    constructor(iframe) {
        this.iframe = iframe;
        this.isReady = false;
        this.messageQueue = [];
        this.eventHandlers = new Map();
        
        this.setupMessageHandling();
    }
    
    setupMessageHandling() {
        window.addEventListener('message', (event) => {
            if (event.source === this.iframe.contentWindow) {
                this.handleMessage(event.data);
            }
        });
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'viewer_ready':
                this.isReady = true;
                this.processMessageQueue();
                this.emit('ready', data);
                break;
                
            case 'viewer_event':
                this.emit(data.event, data.data);
                break;
                
            default:
                this.emit('message', data);
        }
    }
    
    sendMessage(message) {
        if (!this.isReady) {
            this.messageQueue.push(message);
            return;
        }
        
        if (this.iframe && this.iframe.contentWindow) {
            this.iframe.contentWindow.postMessage(message, '*');
        }
    }
    
    processMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.sendMessage(message);
        }
    }
    
    sendData(data) {
        this.sendMessage({
            type: 'hunyuan_data',
            ...data
        });
    }
    
    sendCommand(command, params = {}) {
        this.sendMessage({
            type: 'viewer_command',
            command: command,
            ...params
        });
    }
    
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }
    
    off(event, handler) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }
    
    emit(event, data) {
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }
}

/**
 * Viewer widget utilities
 */
export class ViewerWidget {
    /**
     * Create a 3D viewer widget for ComfyUI nodes
     */
    static create(node, options = {}) {
        const {
            width = 800,
            height = 600,
            showControls = true,
            showInfo = true,
            autoResize = true
        } = options;
        
        // Create container
        const container = document.createElement('div');
        container.className = 'hunyuan-3d-viewer hunyuan-viewer-fade-in';
        container.style.width = width + 'px';
        container.style.height = height + 'px';
        
        // Create iframe
        const iframe = document.createElement('iframe');
        iframe.src = '/custom_nodes/ComfyUI_HunyuanWorld/web/enhanced_viewer.html';
        iframe.style.width = '100%';
        iframe.style.height = '100%';
        iframe.style.border = 'none';
        
        // Create loading overlay
        const loading = document.createElement('div');
        loading.className = 'hunyuan-viewer-loading';
        loading.innerHTML = '<div class="spinner"></div>Loading 3D viewer...';
        
        container.appendChild(iframe);
        container.appendChild(loading);
        
        // Setup communication
        const communication = new ViewerCommunication(iframe);
        
        // Hide loading when ready
        communication.on('ready', () => {
            loading.style.display = 'none';
        });
        
        // Handle errors
        communication.on('error', (error) => {
            ViewerWidget.showError(container, error.message);
        });
        
        // Auto-resize if enabled
        if (autoResize) {
            const resizeObserver = new ResizeObserver(() => {
                communication.sendCommand(VIEWER_COMMANDS.RESIZE);
            });
            resizeObserver.observe(container);
        }
        
        // Store references
        container.viewerIframe = iframe;
        container.communication = communication;
        container.node = node;
        
        return container;
    }
    
    /**
     * Show error in viewer
     */
    static showError(container, message) {
        const existing = container.querySelector('.hunyuan-viewer-error');
        if (existing) {
            existing.remove();
        }
        
        const error = document.createElement('div');
        error.className = 'hunyuan-viewer-error';
        error.innerHTML = `
            <div>
                <div class="error-icon">⚠️</div>
                <div>Viewer Error: ${message}</div>
            </div>
        `;
        
        container.appendChild(error);
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (error.parentNode) {
                error.remove();
            }
        }, 5000);
    }
    
    /**
     * Show loading state
     */
    static showLoading(container, message = 'Loading...') {
        let loading = container.querySelector('.hunyuan-viewer-loading');
        if (!loading) {
            loading = document.createElement('div');
            loading.className = 'hunyuan-viewer-loading';
            container.appendChild(loading);
        }
        
        loading.innerHTML = `<div class="spinner"></div>${message}`;
        loading.style.display = 'flex';
    }
    
    /**
     * Hide loading state
     */
    static hideLoading(container) {
        const loading = container.querySelector('.hunyuan-viewer-loading');
        if (loading) {
            loading.style.display = 'none';
        }
    }
}

/**
 * Data validation utilities
 */
export class DataValidator {
    /**
     * Validate panorama image data
     */
    static validatePanoramaData(data) {
        const errors = [];
        
        if (!data) {
            errors.push('Panorama data is required');
            return errors;
        }
        
        if (!data.image && !data.imageUrl && !data.imageTensor) {
            errors.push('Panorama image, URL, or tensor is required');
        }
        
        if (data.imageTensor) {
            // Validate aspect ratio for panoramic images (should be ~2:1)
            if (data.metadata && data.metadata.width && data.metadata.height) {
                const aspectRatio = data.metadata.width / data.metadata.height;
                if (Math.abs(aspectRatio - 2.0) > 0.5) {
                    errors.push(`Panorama aspect ratio ${aspectRatio.toFixed(2)} may not be equirectangular (expected ~2.0)`);
                }
            }
        }
        
        return errors;
    }
    
    /**
     * Validate 3D mesh data
     */
    static validateMeshData(data) {
        const errors = [];
        
        if (!data) {
            errors.push('Mesh data is required');
            return errors;
        }
        
        if (!data.vertices || !Array.isArray(data.vertices)) {
            errors.push('Mesh vertices array is required');
        } else if (data.vertices.length === 0) {
            errors.push('Mesh must have at least one vertex');
        }
        
        if (!data.faces || !Array.isArray(data.faces)) {
            errors.push('Mesh faces array is required');
        } else if (data.faces.length === 0) {
            errors.push('Mesh must have at least one face');
        }
        
        // Validate vertex format
        if (data.vertices && data.vertices.length > 0) {
            const firstVertex = data.vertices[0];
            if (!Array.isArray(firstVertex) || firstVertex.length !== 3) {
                errors.push('Vertices must be arrays of 3 numbers [x, y, z]');
            }
        }
        
        // Validate face format
        if (data.faces && data.faces.length > 0) {
            const firstFace = data.faces[0];
            if (!Array.isArray(firstFace) || firstFace.length !== 3) {
                errors.push('Faces must be arrays of 3 vertex indices');
            }
        }
        
        // Validate face indices
        if (data.vertices && data.faces) {
            const maxVertexIndex = data.vertices.length - 1;
            for (let i = 0; i < Math.min(data.faces.length, 10); i++) {
                const face = data.faces[i];
                if (Array.isArray(face)) {
                    for (const index of face) {
                        if (index < 0 || index > maxVertexIndex) {
                            errors.push(`Face ${i} references invalid vertex index ${index}`);
                            break;
                        }
                    }
                }
            }
        }
        
        return errors;
    }
    
    /**
     * Validate layered scene data
     */
    static validateLayeredSceneData(data) {
        const errors = [];
        
        if (!data) {
            errors.push('Layered scene data is required');
            return errors;
        }
        
        if (data.panorama) {
            errors.push(...this.validatePanoramaData(data.panorama));
        }
        
        if (data.foregroundLayers && Array.isArray(data.foregroundLayers)) {
            data.foregroundLayers.forEach((layer, index) => {
                if (!layer.name) {
                    errors.push(`Layer ${index} missing name`);
                }
                // Add more layer-specific validation as needed
            });
        }
        
        return errors;
    }
}

/**
 * Performance monitoring utilities
 */
export class ViewerPerformance {
    constructor() {
        this.metrics = {
            loadTime: 0,
            renderTime: 0,
            memoryUsage: 0,
            fps: 0,
            triangleCount: 0
        };
        
        this.startTime = performance.now();
    }
    
    recordLoadTime() {
        this.metrics.loadTime = performance.now() - this.startTime;
    }
    
    updateMetrics(newMetrics) {
        Object.assign(this.metrics, newMetrics);
    }
    
    getReport() {
        return {
            ...this.metrics,
            timestamp: Date.now(),
            userAgent: navigator.userAgent,
            hardwareConcurrency: navigator.hardwareConcurrency,
            deviceMemory: navigator.deviceMemory
        };
    }
}

/**
 * Export all utilities
 */
export default {
    DATA_TYPES,
    VIEWER_COMMANDS,
    VIEW_MODES,
    RENDER_QUALITY,
    DataConverter,
    ViewerCommunication,
    ViewerWidget,
    DataValidator,
    ViewerPerformance
};