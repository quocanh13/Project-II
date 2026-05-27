import { type ChangeEvent, useEffect, useRef } from "react";
import { useFaceDetectionStore } from "../store/faceDetectionStore";
import { drawBboxesOnCanvas } from "../utils/image";
import addImageIcon from "../assets/default/add_image.png";
import { detectFace } from "../api/faceDetectionAPI";
import { useToastStore } from "../store/layoutStore";
import type { FaceDetectionBackbone } from "../types";
import "../styles/FaceDetection.css";

function FaceDetection() {
    return (
        <div className="face-detection">
            <div className="face-detection-title">
                <h1>Face <span>Detection</span></h1>
                <span className="face-detection-title-badge">Beta</span>
            </div>
            <div className="face-detection-content">
                <AddImage />
                <ResultImage />
            </div>
            <DetectionActions />
        </div>
    );
}

function AddImage() {
    const { 
        setImage, setImageURL, 
        imageURL, setResultImageURL, 
        setDetectionResult
    } = useFaceDetectionStore();
    const inputRef = useRef<HTMLInputElement>(null);

    const onAddImage = async (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] || null;
        setImage(file);
        if(file != null){
            const imageURL = URL.createObjectURL(file)
            setImageURL(imageURL)
            setResultImageURL(imageURL)
            setDetectionResult(null)
        }
    };

    return (
        <div className="add-image-wrapper">
            <div className="add-image-card">
                <span className="add-image-label">Source Image</span>
                <div 
                    className={`add-image-zone ${imageURL ? 'has-image' : ''}`}
                    onClick={() => inputRef.current?.click()}
                >
                    {imageURL ? (
                        <>
                            <img src={imageURL} alt="Preview" className="add-image-preview" />
                            <div className="add-image-overlay">
                                <span className="add-image-overlay-text">Change Image</span>
                            </div>
                        </>
                    ) : (
                        <div className="add-image-placeholder">
                            <img src={addImageIcon} className="add-image-placeholder-icon" alt="icon" />
                            <p className="add-image-placeholder-text">
                                <strong>Click to upload</strong> or drag and drop
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

function ResultImage() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { 
        imageURL, resultImageURL, detectionResult,bboxColor,        
        bboxLineWidth, numBbox     
    } = useFaceDetectionStore();

    useEffect(()=>{
        if(canvasRef.current != null) {
            const bboxs = detectionResult == null ? undefined : detectionResult.map((v) => {
                return v.bbox
            })
            drawBboxesOnCanvas(canvasRef.current, resultImageURL, undefined, undefined, bboxs, Number(numBbox), bboxLineWidth, bboxColor)
        }

    },[resultImageURL, detectionResult, bboxColor, bboxLineWidth, numBbox]) 

    return (
        <div className="result-image-wrapper">
            <div className="result-image-card">
                <span className="result-image-label">Detection Result</span>
                <div className={`result-image-zone ${imageURL ? 'has-result' : ''}`}>
                    {imageURL ? (
                        <canvas ref={canvasRef} className="result-canvas"></canvas>
                    ) : (
                        <div className="result-image-placeholder">
                            <p>Waiting for detection...</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function DetectionActions() {
    const {
        setDetectionResult,
        image,
        algorithm, setAlgorithm,
        backbone, setBackbone, 
        dataset, setDataset,
        numBbox, setNumBbox,
        bboxColor, setBboxColor,           
        bboxLineWidth, setBboxLineWidth,   
    } = useFaceDetectionStore()
    const {addToast} = useToastStore()

    function onChangeBackBone(e: ChangeEvent<HTMLSelectElement>){
        if(e.target.value) setBackbone(e.target.value as FaceDetectionBackbone)
    }

    function onChangeAlgorithm(e: ChangeEvent<HTMLSelectElement>){
        if(e.target.value) setAlgorithm(e.target.value)
    }

    function onChangeDataset(e: ChangeEvent<HTMLSelectElement>){
        if(e.target.value) setDataset(e.target.value)
    }

    function onChangeColor(e: ChangeEvent<HTMLSelectElement>){
        if(e.target.value) setBboxColor(e.target.value)
    }

    function onChangeThickness(e: ChangeEvent<HTMLInputElement>) {
        const val = parseInt(e.target.value, 10);
        if (!isNaN(val) && val >= 1) {
            setBboxLineWidth(val);
        } else if (e.target.value === "") {
            setBboxLineWidth(0);
        }
    }

    async function onClickDetect(){
        if(image == null){
            addToast("Face Detection", "You haven't added an image")
            return
        }
        const res = await detectFace(image, {backbone, algorithm, dataset}, Number(numBbox))
        if(res.error) {
            if(res.message != undefined)
                addToast("Face Detection", res.message)
            return
        }
        
        const detectionResult = res.result
        if(detectionResult == undefined) {
            addToast("Face Detection", "Server error. There is no DetectionResult")
            return
        }
        
        setDetectionResult(detectionResult)
        addToast("Detection Successully", "Face Detection has finished", undefined, false)
    }
    return (
        <div className="detection-panel">
            <div className="detection-card">
                <div className="detection-controls">
                    <div className="control-group">
                        <label className="control-label">Backbone Model</label>
                        <select className="control-select" value={backbone} onChange={onChangeBackBone}>
                            <option value="resnet34">ResNet-34</option>
                            <option value="resnet18">ResNet-18</option>
                            <option value="vgg16">VGG-16</option>
                        </select>
                    </div>

                    <div className="control-group">
                        <label className="control-label">Algorithm</label>
                        <select className="control-select" value={algorithm} onChange={onChangeAlgorithm}>
                            <option value="faster_rcnn">Faster R-CNN</option>
                        </select>
                    </div>

                    <div className="control-group">
                        <label className="control-label">Dataset</label>
                        <select className="control-select" value={dataset} onChange={onChangeDataset}>
                            <option value="wider_face">WIDER FACE</option>
                            <option value="celeb_A">CelebA</option>
                        </select>
                    </div>

                    <div className="control-group">
                        <label className="control-label">Number of BBox</label>
                        <input 
                            type="number" 
                            className="control-input" 
                            value={numBbox.toString()}
                            min={1}
                            placeholder="Enter number"
                            onChange={(e) => {
                                setNumBbox(e.target.value);
                            }}
                        />
                    </div>

                    {/* Phần chọn màu đã chuyển đổi thành Dropdown chọn tên màu đơn giản */}
                    <div className="control-group">
                        <label className="control-label">BBox Color</label>
                        <select className="control-select" value={bboxColor ?? "red"} onChange={onChangeColor}>
                            <option value="red">Red</option>
                            <option value="blue">Blue</option>
                            <option value="green">Green</option>
                            <option value="black">Black</option>
                            <option value="white">White</option>
                        </select>
                    </div>

                    <div className="control-group">
                        <label className="control-label">BBox Thickness (px)</label>
                        <input 
                            type="number" 
                            className="control-input" 
                            value={bboxLineWidth.toString() ?? 2}
                            min={1}
                            max={10}
                            placeholder="e.g. 2"
                            onChange={onChangeThickness}
                        />
                    </div>
                </div>

                <button className="detect-button" onClick={onClickDetect}>Detect Face</button>
            </div>
        </div>
    );
}

export default FaceDetection;