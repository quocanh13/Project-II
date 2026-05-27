import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./Layout";
import FaceDetection from "./FaceDetection";
import FaceComparison from "./FaceComparison";
import { FaceLandmark } from "./FaceLandmark";
import {FaceVerification} from "./FaceVerification";
import '../styles/App.css';

function App() {
    return (
        <div id="center">
            <BrowserRouter>
                <Routes>
                    <Route path="/" element={<Layout />}>
                        <Route index element={<FaceDetection />} />
                        <Route path="face_detection" element={<FaceDetection />} />
                        <Route path="face_landmark" element={<FaceLandmark/>}></Route>
                        <Route path="face_comparison" element={<FaceComparison/>}></Route>
                        <Route path="face_verification" element={<FaceVerification/>}></Route>
                    </Route>
                </Routes>
            </BrowserRouter>
        </div>
    );
}

export default App;