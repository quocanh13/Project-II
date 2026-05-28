import { type ChangeEvent, useEffect, useRef } from "react";
import { drawLandmarkOnCanvas } from "../utils/image"; 
import addImageIcon from "../assets/default/add_image.png";
import { useToastStore } from "../store/layoutStore";
import { useFaceLandmarkStore } from "../store/faceLandmarkStore";
import { detectLandmark } from "../api/faceLandmarkAPI";
import { type FaceLandmarkBackbone } from "../types";
import "../styles/FaceLandmark.css";

export function FaceLandmark() {
    return (
        <div className="face-landmark">
            <div className="face-landmark-title">
                <h1>Face <span>Landmark</span></h1>
                <span className="face-landmark-title-badge">Beta</span>
            </div>
            <div className="face-landmark-content">
                <AddLandmarkImage />
                <ResultLandmarkImage />
            </div>
            <LandmarkActions />
        </div>
    );
}

function AddLandmarkImage() {
    const { 
        setImage, setImageURL, 
        imageURL, setResultImageURL, 
        setLandmarkResult
    } = useFaceLandmarkStore();
    const { addToast } = useToastStore();
    const inputRef = useRef<HTMLInputElement>(null);

    const onAddImage = async (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] || null;
        setImage(file);
        if (file != null) {
            const imageURL = URL.createObjectURL(file);
            setImageURL(imageURL);
            setResultImageURL(imageURL);
            setLandmarkResult(null);
        }
    };

    return (
        <div className="landmark-image-wrapper">
            <div className="landmark-image-card">
                <span className="landmark-image-label">Source Image</span>
                <div 
                    className={`landmark-image-zone ${imageURL ? 'has-image' : ''}`}
                    onClick={() => inputRef.current?.click()}
                >
                    {imageURL ? (
                        <>
                            <img src={imageURL} alt="Preview" className="landmark-image-preview" />
                            <div className="landmark-image-overlay">
                                <span className="landmark-image-overlay-text">Change Image</span>
                            </div>
                        </>
                    ) : (
                        <div className="landmark-image-placeholder">
                            <img src={addImageIcon} className="landmark-image-placeholder-icon" alt="icon" />
                            <p className="landmark-image-placeholder-text">
                                <strong>Click to upload</strong> or drag and drop
                            </p>
                        </div>
                    )}
                    <input 
                        ref={inputRef}
                        type="file" 
                        className="landmark-image-input" 
                        onChange={onAddImage} 
                        accept="image/*" 
                    />
                </div>
            </div>
        </div>
    );
}

function ResultLandmarkImage() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { imageURL, resultImageURL, landmarkResult, radius } = useFaceLandmarkStore();

    useEffect(() => {
        if (canvasRef.current != null) {
            if(imageURL != null) {
                drawLandmarkOnCanvas(canvasRef.current, imageURL, undefined, undefined, landmarkResult?.landmark, radius)
            }
        }
    }, [resultImageURL, landmarkResult, radius]);

    return (
        <div className="landmark-result-wrapper">
            <div className="landmark-result-card">
                <span className="landmark-result-label">Landmark Result</span>
                <div className={`landmark-result-zone ${imageURL ? 'has-result' : ''}`}>
                    {imageURL ? (
                        <canvas ref={canvasRef} className="landmark-result-canvas"></canvas>
                    ) : (
                        <div className="landmark-result-placeholder">
                            <p>Waiting for landmark detection...</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function LandmarkActions() {
    const {
        setLandmarkResult,
        image,
        backbone, setBackbone, 
        radius, setRadius
    } = useFaceLandmarkStore();
    const { addToast } = useToastStore();

    function onChangeBackbone(e: ChangeEvent<HTMLSelectElement>) {
        if (e.target.value) setBackbone(e.target.value);
    }

    function onChangeRadius(e: ChangeEvent<HTMLInputElement>) {
        const val = parseInt(e.target.value, 10);
        if (!isNaN(val) && val >= 0) {
            setRadius(val);
        } else if (e.target.value === "") {
            setRadius(0);
        }
    }

    async function onClickDetect(){
        if(image == null){
            addToast("Image Error", "You need to upload an image", 5000)
            return
        }
        const res = await detectLandmark(image, {backbone: backbone as FaceLandmarkBackbone})
        if(res.error){
            addToast("Server Error", res.message, 5000)
        } else {
            if(res.result != null){
                setLandmarkResult(res.result)
            }
        }
    }

    return (
        <div className="landmark-panel">
            <div className="landmark-card">
                <div className="landmark-controls">
                    <div className="landmark-control-group">
                        <label className="landmark-control-label">Backbone Model</label>
                        <select className="landmark-control-select" value={backbone} onChange={onChangeBackbone}>
                            <option value="resnet34">ResNet-34</option>
                            <option value="resnet18">ResNet-18</option>
                        </select>
                    </div>
                    <div className="landmark-control-group">
                        <label className="landmark-control-label">Landmark Radius</label>
                        <input 
                            type="number" 
                            className="landmark-control-input"
                            min={0}
                            value={radius.toString() ?? 0}
                            onChange={onChangeRadius}
                            placeholder="e.g. 4"
                        />
                    </div>
                </div>

                <button className="landmark-detect-button" onClick={onClickDetect}>Detect Landmarks</button>
            </div>
        </div>
    );
}