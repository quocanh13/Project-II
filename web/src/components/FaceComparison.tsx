import { type ChangeEvent, useEffect, useRef } from "react";
import addImageIcon from "../assets/default/add_image.png";
import { useFaceComparisonStore } from "../store/faceComparisonStore";
import { drawBboxesOnCanvas } from "../utils/image"; // Sử dụng hàm vẽ đồng bộ từ utils
import { compareFace } from "../api/faceComparisonAPI";
import { useToastStore } from "../store/layoutStore";
import type { FaceDetectionAlgorithm, FaceDetectionBackbone, FaceDetectionDataset, FaceEmbedderBackbone, FaceLandmarkBackbone } from "../types";
import "../styles/FaceComparison.css";

function FaceComparison() {
    return (
        <div className="face-comparison">
            <div className="face-comparison-title">
                <h1>Face <span>Comparison</span></h1>
                <span className="face-comparison-title-badge">Beta</span>
            </div>
            
            <div className="face-comparison-content">
                {[0, 1].map(v => <AddImage id={v} key={v} />)}
            </div>

            <ComparisonActions />
        </div>
    );
}

function AddImage({ id }: { id: number }) {
    const { images, setImages, comparisonResult } = useFaceComparisonStore();
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (images[id] == null || canvasRef.current == null) return;
        const imgURL = URL.createObjectURL(images[id]);
        drawBboxesOnCanvas(canvasRef.current, imgURL, undefined, undefined, undefined);

        return () => {
            URL.revokeObjectURL(imgURL);
        };
    }, [images[id], comparisonResult]);

    const onAddImage = (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] || null;
        if (file != null) {
            images[id] = file
            setImages(images)
        }
    };

    return (
        <div className="add-image-wrapper">
            <div className="add-image-card">
                <span className="add-image-label">Source Image {id + 1}</span>
                <div 
                    className={`add-image-zone ${images[id] ? 'has-image' : ''}`}
                    onClick={() => inputRef.current?.click()}
                >
                    {images[id] ? (
                        <>
                            <canvas ref={canvasRef} className="result-canvas" />
                            <div className="add-image-overlay">
                                <span className="add-image-overlay-text">Change Image</span>
                            </div>
                        </>
                    ) : (
                        <div className="add-image-placeholder">
                            <img src={addImageIcon} className="add-image-placeholder-icon" alt="icon" />
                            <p className="add-image-placeholder-text">
                                <strong>Click to upload</strong> image {id + 1}
                            </p>
                        </div>
                    )}
                    <input 
                        ref={inputRef}
                        type="file" 
                        className="add-image-input" 
                        onChange={onAddImage} 
                        accept="image/*" 
                    />
                </div>
            </div>
        </div>
    );
}

function ComparisonActions() {
    const {
        getValidImages,
        faceDetectionConfig, setFaceDetectionConfig,
        faceLandmarkConfig, setFaceLandmarkConfig,
        faceEmbedderConfig, setFaceEmbedderConfig,
        comparisonResult, setComparisonResult
    } = useFaceComparisonStore();

    const { addToast } = useToastStore();

    async function onClickCompare() {
        const images = getValidImages();
        if (images == null || images.length < 2) {
            addToast("Face Comparison", "You have to add two images to compare", 3000);
            return;
        }
        const res = await compareFace(faceDetectionConfig, faceLandmarkConfig, faceEmbedderConfig, images)
        if (res.error) {
            if (res.message != undefined)
                addToast("Face Comparison", res.message, 3000);
            return;
        }

        const comparisonResult = res.result;
        if (comparisonResult == undefined) {
            addToast("Server Error", "There is no Detection Result");
            return;
        }
        setComparisonResult({...comparisonResult});
        if(!comparisonResult.frontals[0]) addToast("Warning", "The face in the image 1 is not frontal. That can cause wrong result")
        if(!comparisonResult.frontals[1]) addToast("Warning", "The face in the image 2 is not frontal. That can cause wrong result")
        console.log(comparisonResult)
        addToast( "Compare Successfully", "Face Comparison Has Finished", undefined, false);
    }

    return (
        <div className="comparison-panel">
            <div className="comparison-card">
                
                {/* Section 1: Face Detection */}
                <div className="control-section-group">
                    <h3 className="control-section-title">1. Face Detection Config</h3>
                    <div className="comparison-controls column-3">
                        <div className="control-group">
                            <label className="control-label">Algorithm</label>
                            <select 
                            className="control-select" 
                            value={faceDetectionConfig.algorithm} 
                            onChange={(e) => 
                                setFaceDetectionConfig({...faceDetectionConfig, algorithm : e.target.value as FaceDetectionAlgorithm})
                            }>
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
                            }>
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
                            }>
                                <option value="wider_face">WIDER FACE</option>
                                <option value="celeb_a">CelebA</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Section 2 & 3: Face Landmark & Embedder */}
                <div className="control-section-group">
                    <h3 className="control-section-title">2 & 3. Alignment & Feature Extraction</h3>
                    <div className="comparison-controls column-2">
                        <div className="control-group">
                            <label className="control-label">Face Landmark Backbone</label>
                            <select 
                            className="control-select" 
                            value={faceLandmarkConfig.backbone} 
                            onChange={(e) => setFaceLandmarkConfig({...faceLandmarkConfig, backbone: e.target.value as FaceLandmarkBackbone})}>
                                <option value="resnet18">ResNet-18</option>
                                <option value="resnet34">ResNet-34</option>
                            </select>
                        </div>
                        <div className="control-group">
                            <label className="control-label">Face Embedder Backbone</label>
                            <select 
                            className="control-select" 
                            value={faceEmbedderConfig.backbone} 
                            onChange={(e) => setFaceEmbedderConfig({...faceEmbedderConfig, backbone: e.target.value as FaceEmbedderBackbone})}>
                                <option value="resnet18">ResNet-18</option>
                                <option value="resnet34">ResNet-34</option>
                                <option value="resnet50">ResNet-50</option>
                                <option value="vgg16">VGG-16</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Section 4: Results Output */}
                <div className="control-section-group">
                    <h3 className="control-section-title">Metrics Output</h3>
                    <div className="comparison-controls column-1">
                        <div className="control-group">
                            <label className="control-label">Calculated Distance</label>
                            <input 
                                type="text" 
                                className="control-input" 
                                value={comparisonResult?.distance} 
                                readOnly 
                                style={{ backgroundColor: '#f3f4f6', cursor: 'not-allowed' }}
                            />
                        </div>
                    </div>
                </div>

                <button className="compare-button" onClick={onClickCompare}>Compare Faces</button>
                
                <div className="confidence-section">
                    <div className="confidence-label">
                        <span>Similarity Score</span>
                        <span className="confidence-value">{comparisonResult == undefined ? -1 : comparisonResult.percent*100}</span>
                    </div>
                    <div className="confidence-bar">
                        <div 
                        className="confidence-fill" 
                        style={{ width: `${comparisonResult == undefined? 0 : comparisonResult.percent*100}%` }}></div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default FaceComparison;