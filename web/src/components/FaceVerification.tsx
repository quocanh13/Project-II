import { useEffect, useRef } from "react";
import { useFaceVerificationStore } from "../store/faceVerificationStore";
import type { FaceDetectionAlgorithm, FaceDetectionBackbone, FaceDetectionConfig, FaceDetectionDataset, FaceEmbedderBackbone, FaceLandmarkBackbone, FaceLandmarkConfig } from "../types";
import "../styles/FaceVerification.css";
import { drawBboxesOnCanvas, drawLandmarkOnCanvas } from "../utils/image";
import { sampleFace as sampleFaceAPI, verifyFace as verifyFaceAPI } from "../api/faceVerificationAPI";
import { useToastStore } from "../store/layoutStore";

export function FaceVerification() {
    return (
        <div className="face-verification">
            <div className="face-verification-title">
                <h1>Face <span>Verification</span></h1>
                <span className="face-verification-title-badge">Beta</span>
            </div>
            
            <div className="face-verification-content">
                <FaceCapture />
            </div>

            <VerificationAction />
        </div>
    );
}

export function FaceCapture() {
    const {
        verificationResult, setVerificationResult,
        getDone, setDone,
        getMode, setMode,
        faceDetectionConfig, faceLandmarkConfig, faceEmbedderConfig,
    } = useFaceVerificationStore()
    const {
        addToast
    } = useToastStore()
    const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);
    const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    async function sampleFace(
        image: Blob | null, 
    ){
        if(image != null){
            const res = await sampleFaceAPI(faceDetectionConfig, faceLandmarkConfig, image)
            setDone(true)
            if(res.error){
                addToast("Error", res.message)
                return;
            }
            if(res.result != null){
                const result = res.result
                console.log(res.result)
                setVerificationResult(result)
                if(!result.frontal){
                    addToast("Invalid Image", "Face is not frontal")
                    return
                }
                if(result.ok){
                    addToast("Sample Successfully", "Sample successfully, switched to verify mode", 5000, false)
                    setMode("verify")
                }
            }
        }
    }

    async function verifyFace(
        image: Blob | null, 
    ){
        if(image != null){
            const res = await verifyFaceAPI(faceDetectionConfig, faceLandmarkConfig, faceEmbedderConfig, image)
            setDone(true)
            if(res.error){
                addToast("Error", res.message)
                return;
            }
            if(res.result != null){
                const result = res.result
                console.log(res.result)
                setVerificationResult(result)
                if(!result.frontal) addToast("Invalid Image", "Face is not frontal", 4000)
            }
        }
    }

    useEffect(()=>{
        const bbox = verificationResult?.detection.map((v)=>v.bbox)
        const color = verificationResult?.ok? "green" : "red" 
        if(overlayCanvasRef.current != null){
            drawBboxesOnCanvas(
                overlayCanvasRef.current, undefined, 
                videoRef.current?.videoWidth, videoRef.current?.videoHeight, 
                bbox, -1, 3, color
            )
            drawLandmarkOnCanvas(
                overlayCanvasRef.current, undefined, 
                videoRef.current?.videoWidth, videoRef.current?.videoHeight,
                verificationResult?.landmark
            )
        }
    }, [verificationResult])

    function captureVideo(){
        const video = videoRef.current;
        if (!video) return;

        if(!getDone()){
            video.requestVideoFrameCallback(captureVideo)
            return
        }
        setDone(false)
        const hiddenCanvas = hiddenCanvasRef.current;
        if (hiddenCanvas) {
            const hiddenCtx = hiddenCanvas.getContext("2d");
            hiddenCtx?.drawImage(video, 0, 0);
            hiddenCanvas.toBlob(getMode() == "sample"? sampleFace : verifyFace)
        }
        video.requestVideoFrameCallback(captureVideo)
    }

    useEffect(()=>{
        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({video : {width: 1920, height: 1080}, audio: false});
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.onloadedmetadata = () => {

                    const hiddenCanvas = hiddenCanvasRef.current;
                        if (hiddenCanvas) {
                            hiddenCanvas.width = videoRef.current!.videoWidth;
                            hiddenCanvas.height = videoRef.current!.videoWidth;
                        }

                    const overlayCanvas = overlayCanvasRef.current;
                    if (overlayCanvas) {
                        overlayCanvas.width = videoRef.current!.videoWidth;
                        overlayCanvas.height = videoRef.current!.videoWidth;
                    }
                    captureVideo();
                    };
                }
            } catch (err) {
                console.error("Error accessing webcam: ", err);
            }
        }
        startCamera();
    }, []);

    return (
        <div className="capture-image-wrapper">
            <div className="capture-image-card">
                <span className="capture-image-label">Live Camera Feed</span>
                <div className="capture-image-zone has-image">
                    <canvas ref={hiddenCanvasRef} className="hidden-canvas" style={{ display: 'none' }}/>
                    
                    <div className="video-container">
                        <video ref={videoRef} className="video-preview" autoPlay playsInline muted></video>
                        
                        <canvas ref={overlayCanvasRef} className="overlay-canvas"/>
                    </div>
                </div>
            </div>
        </div>
    );
};

export function VerificationAction() {
    const {
        mode, setMode, 
        faceDetectionConfig, setFaceDetectionConfig,
        faceLandmarkConfig, setFaceLandmarkConfig,
        faceEmbedderConfig, setFaceEmbedderConfig,
    } = useFaceVerificationStore();

    return (
        <div className="verification-panel">
            <div className="verification-card">
                
                <div className="control-section-group">
                    <h3 className="control-section-title">Execution Mode</h3>
                    <div className="verification-controls column-1">
                        <div className="control-group">
                            <label className="control-label">Select Mode</label>
                            <select 
                                className="control-select mode-selector" 
                                value={mode} 
                                onChange={(e) => setMode(e.target.value as "sample" | "verify")}
                            >
                                <option value="sample">Sample Mode</option>
                                <option value="verify">Verify Mode</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div className="control-section-group">
                    <h3 className="control-section-title">1. Face Detection Config</h3>
                    <div className="verification-controls column-3">
                        <div className="control-group">
                            <label className="control-label">Algorithm</label>
                            <select 
                                className="control-select" 
                                value={faceDetectionConfig.algorithm} 
                                onChange={(e) => 
                                    setFaceDetectionConfig({...faceDetectionConfig, algorithm : e.target.value as FaceDetectionAlgorithm})
                                }
                            >
                                <option value="faster_rcnn">Faster R-CNN</option>
                            </select>
                        </div>
                        <div className="control-group">
                            <label className="control-label">Detection Backbone</label>
                            <select 
                                className="control-select" 
                                value={faceDetectionConfig.backbone} 
                                onChange={(e) => 
                                    setFaceDetectionConfig({...faceDetectionConfig, backbone: e.target.value as FaceDetectionBackbone})  
                                }
                            >
                                <option value="vgg16">VGG-16</option>
                                <option value="resnet18">ResNet-18</option>
                                <option value="resnet34">ResNet-34</option>
                            </select>
                        </div>
                        <div className="control-group">
                            <label className="control-label">Dataset</label>
                            <select 
                                className="control-select" 
                                value={faceDetectionConfig.dataset} 
                                onChange={(e) => 
                                    setFaceDetectionConfig({...faceDetectionConfig, dataset: e.target.value as FaceDetectionDataset})  
                                }
                            >
                                <option value="wider_face">WIDER FACE</option>
                                <option value="celeb_a">CelebA</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div className="control-section-group">
                    <h3 className="control-section-title">2 & 3. Alignment & Feature Extraction</h3>
                    <div className="verification-controls column-2">
                        <div className="control-group">
                            <label className="control-label">Face Landmark Backbone</label>
                            <select 
                                className="control-select" 
                                value={faceLandmarkConfig.backbone} 
                                onChange={(e) => setFaceLandmarkConfig({...faceLandmarkConfig, backbone: e.target.value as FaceLandmarkBackbone})}
                            >
                                <option value="resnet18">ResNet-18</option>
                                <option value="resnet34">ResNet-34</option>
                            </select>
                        </div>
                        <div className="control-group">
                            <label className="control-label">Face Embedder Backbone</label>
                            <select 
                                className="control-select" 
                                value={faceEmbedderConfig.backbone} 
                                onChange={(e) => setFaceEmbedderConfig({...faceEmbedderConfig, backbone: e.target.value as FaceEmbedderBackbone})}
                            >
                                <option value="resnet18">ResNet-18</option>
                                <option value="resnet34">ResNet-34</option>
                                <option value="resnet50">ResNet-50</option>
                                <option value="vgg16">VGG-16</option>
                            </select>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}



export default FaceVerification;