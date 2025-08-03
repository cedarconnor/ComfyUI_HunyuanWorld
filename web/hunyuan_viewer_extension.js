// ComfyUI Web Extension for HunyuanWorld 3D Viewer
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Register the enhanced 3D viewer widget
app.registerExtension({
    name: "HunyuanWorld.3DViewer",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add 3D viewer widget to HunyuanViewer nodes
        if (nodeData.name === "HunyuanViewer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Create the 3D viewer widget
                this.addWidget("html", "3d_viewer", "", (value, widget, node) => {
                    // Widget callback - not needed for display widget
                }, {
                    serialize: false,
                    hideOnZoom: false
                });
                
                // Add viewer widget after creation
                this.setupViewer = () => {
                    this.setup3DViewer();
                };
                
                return result;
            };
            
            // Setup 3D viewer functionality
            nodeType.prototype.setup3DViewer = function() {
                console.log("Setting up HunyuanWorld 3D viewer for node:", this.id);
                
                // Create iframe for 3D viewer
                const iframe = document.createElement("iframe");
                iframe.src = `/custom_nodes/ComfyUI_HunyuanWorld/web/enhanced_viewer.html`;
                iframe.style.width = "800px";
                iframe.style.height = "600px";
                iframe.style.border = "1px solid #333";
                iframe.style.borderRadius = "8px";
                iframe.style.backgroundColor = "#000";
                
                // Store iframe reference
                this.viewerIframe = iframe;
                
                // Setup message handling
                this.setupViewerCommunication();
                
                // Add iframe to widget
                const viewerWidget = this.widgets.find(w => w.name === "3d_viewer");
                if (viewerWidget) {
                    viewerWidget.element = iframe;
                    viewerWidget.computeSize = () => [800, 600];
                }
                
                return iframe;
            };
            
            // Setup communication with 3D viewer
            nodeType.prototype.setupViewerCommunication = function() {
                // Listen for messages from viewer
                window.addEventListener("message", (event) => {
                    if (event.source === this.viewerIframe.contentWindow) {
                        this.handleViewerMessage(event.data);
                    }
                });
                
                // Setup data sending when viewer is ready
                this.viewerReady = false;
                this.pendingData = null;
            };
            
            // Handle messages from 3D viewer
            nodeType.prototype.handleViewerMessage = function(data) {
                console.log("Received message from 3D viewer:", data);
                
                switch(data.type) {
                    case 'viewer_ready':
                        console.log("3D viewer is ready");
                        this.viewerReady = true;
                        
                        // Send any pending data
                        if (this.pendingData) {
                            this.sendDataToViewer(this.pendingData);
                            this.pendingData = null;
                        }
                        break;
                        
                    case 'viewer_event':
                        this.handleViewerEvent(data);
                        break;
                }
            };
            
            // Handle viewer events
            nodeType.prototype.handleViewerEvent = function(data) {
                switch(data.event) {
                    case 'view_changed':
                        console.log("Viewer view changed:", data.data);
                        break;
                        
                    case 'export_complete':
                        console.log("Export completed:", data.data);
                        // Could trigger a ComfyUI notification here
                        break;
                        
                    case 'error':
                        console.error("Viewer error:", data.data.message);
                        // Could show error in ComfyUI UI
                        break;
                }
            };
            
            // Send data to 3D viewer
            nodeType.prototype.sendDataToViewer = function(data) {
                if (!this.viewerIframe || !this.viewerReady) {
                    // Store data to send when viewer is ready
                    this.pendingData = data;
                    return;
                }
                
                console.log("Sending data to 3D viewer:", data.dataType);
                
                this.viewerIframe.contentWindow.postMessage({
                    type: 'hunyuan_data',
                    ...data
                }, '*');
            };
            
            // Send commands to viewer
            nodeType.prototype.sendViewerCommand = function(command, params = {}) {
                if (!this.viewerIframe || !this.viewerReady) {
                    console.warn("Viewer not ready for commands");
                    return;
                }
                
                this.viewerIframe.contentWindow.postMessage({
                    type: 'viewer_command',
                    command: command,
                    ...params
                }, '*');
            };
            
            // Override onExecuted to handle new data
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                const result = onExecuted?.apply(this, arguments);
                
                // Process HunyuanWorld data for 3D viewer
                if (message && this.viewerIframe) {
                    this.processExecutionResults(message);
                }
                
                return result;
            };
            
            // Process execution results for 3D display
            nodeType.prototype.processExecutionResults = function(message) {
                console.log("Processing execution results for 3D viewer");
                
                // Extract data from node inputs
                const inputData = this.getInputData();
                
                if (inputData) {
                    this.sendDataToViewer(inputData);
                }
            };
            
            // Get input data for 3D viewer
            nodeType.prototype.getInputData = function() {
                const inputs = this.inputs;
                if (!inputs || inputs.length === 0) {
                    return null;
                }
                
                // Get the input data (this would be processed by ComfyUI execution)
                const inputData = this.getInputDataForSlot(0); // First input slot
                
                if (!inputData) {
                    return null;
                }
                
                // Process different HunyuanWorld data types
                return this.processHunyuanData(inputData);
            };
            
            // Process HunyuanWorld data types for viewer
            nodeType.prototype.processHunyuanData = function(data) {
                // This would be called with actual data from HunyuanWorld nodes
                // For now, return demo data structure
                
                if (data && data.dataType) {
                    switch(data.dataType) {
                        case 'PANORAMA_IMAGE':
                            return {
                                dataType: 'PANORAMA_IMAGE',
                                imageUrl: data.imageUrl,
                                imageTensor: data.imageTensor,
                                metadata: data.metadata || {}
                            };
                            
                        case 'SCENE_3D':
                            return {
                                dataType: 'SCENE_3D',
                                vertices: data.vertices,
                                faces: data.faces,
                                textures: data.textures,
                                depthMap: data.depthMap,
                                semanticMasks: data.semanticMasks,
                                metadata: data.metadata || {}
                            };
                            
                        case 'WORLD_MESH':
                            return {
                                dataType: 'WORLD_MESH',
                                vertices: data.vertices,
                                faces: data.faces,
                                textureCoords: data.textureCoords,
                                textures: data.textures,
                                materials: data.materials,
                                metadata: data.metadata || {}
                            };
                            
                        case 'LAYERED_SCENE_3D':
                            return {
                                dataType: 'LAYERED_SCENE_3D',
                                panorama: data.panorama,
                                backgroundScene: data.backgroundScene,
                                foregroundLayers: data.foregroundLayers,
                                layerDepthMaps: data.layerDepthMaps,
                                layerMasks: data.layerMasks,
                                objectLabels: data.objectLabels,
                                metadata: data.metadata || {}
                            };
                            
                        case 'LAYER_MESH':
                            return {
                                dataType: 'LAYER_MESH',
                                baseMesh: data.baseMesh,
                                layerMeshes: data.layerMeshes,
                                layerHierarchy: data.layerHierarchy,
                                layerTransforms: data.layerTransforms,
                                layerVisibility: data.layerVisibility,
                                metadata: data.metadata || {}
                            };
                    }
                }
                
                // Return demo data for testing
                return {
                    dataType: 'WORLD_MESH',
                    vertices: this.generateDemoVertices(),
                    faces: this.generateDemoFaces(),
                    textures: [],
                    metadata: {
                        source: 'demo',
                        timestamp: Date.now()
                    }
                };
            };
            
            // Generate demo vertices for testing
            nodeType.prototype.generateDemoVertices = function() {
                const vertices = [];
                for (let i = 0; i < 100; i++) {
                    vertices.push([
                        (Math.random() - 0.5) * 4,
                        (Math.random() - 0.5) * 4,
                        (Math.random() - 0.5) * 4
                    ]);
                }
                return vertices;
            };
            
            // Generate demo faces for testing
            nodeType.prototype.generateDemoFaces = function() {
                const faces = [];
                for (let i = 0; i < 50; i++) {
                    faces.push([
                        Math.floor(Math.random() * 100),
                        Math.floor(Math.random() * 100),
                        Math.floor(Math.random() * 100)
                    ]);
                }
                return faces;
            };
            
            // Cleanup on node removal
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                if (this.viewerIframe) {
                    this.viewerIframe.remove();
                }
                
                const result = onRemoved?.apply(this, arguments);
                return result;
            };
        }
    },
    
    // Add custom CSS for 3D viewer
    async setup() {
        const style = document.createElement("style");
        style.textContent = `
            .hunyuan-3d-viewer {
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            }
            
            .hunyuan-3d-viewer iframe {
                display: block;
                width: 100%;
                height: 100%;
                border: none;
                background: #000;
            }
            
            .comfy-widget-html .hunyuan-3d-viewer {
                margin: 10px 0;
            }
        `;
        document.head.appendChild(style);
    }
});

// Export viewer utilities for other extensions
window.HunyuanViewerUtils = {
    // Utility to create viewer widget for any node
    createViewerWidget: function(node, width = 800, height = 600) {
        const iframe = document.createElement("iframe");
        iframe.src = `/custom_nodes/ComfyUI_HunyuanWorld/web/enhanced_viewer.html`;
        iframe.style.width = width + "px";
        iframe.style.height = height + "px";
        iframe.className = "hunyuan-3d-viewer";
        
        return iframe;
    },
    
    // Utility to send data to any viewer iframe
    sendDataToViewer: function(iframe, data) {
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({
                type: 'hunyuan_data',
                ...data
            }, '*');
        }
    },
    
    // Utility to send commands to any viewer iframe
    sendCommandToViewer: function(iframe, command, params = {}) {
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({
                type: 'viewer_command',
                command: command,
                ...params
            }, '*');
        }
    }
};

console.log("HunyuanWorld 3D Viewer extension loaded successfully");