interface Bbox{
    x1 : number, y1 : number,
    x2 : number, y2 : number
}

export function drawBboxesOnCanvas(
    canvas : HTMLCanvasElement,
    imageURL: string | undefined,
    width: number | undefined,
    height: number | undefined,
    bboxes: Bbox[] | undefined,
    num_bbox: number = -1,
    lineWidth: number = 5,
    color: string = "red",
): void {
    if(imageURL != undefined) {
        const img = new Image()
        img.src = imageURL
        img.onload = () => {
            const ctx = canvas.getContext("2d")!
            canvas.width = img.width
            canvas.height = img.height

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            if(bboxes != undefined) {
                if(num_bbox == -1) 
                    num_bbox = bboxes.length
                for(let i = 0; i < Math.min(num_bbox, bboxes.length); i++){
                    const bbox = bboxes[i]
                    const { x1, y1, x2, y2 } = bbox
                    ctx.strokeStyle = color
                    ctx.lineWidth = lineWidth
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
                }
            }
        }
    } else if(width != undefined && height != undefined) {
        const ctx = canvas.getContext("2d")!
        canvas.width = width
        canvas.height = height

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if(bboxes != undefined) {
            if(num_bbox == -1) 
                num_bbox = bboxes.length
            for(let i = 0; i < Math.min(num_bbox, bboxes.length); i++){
                const bbox = bboxes[i]
                console.log(bbox)
                const { x1, y1, x2, y2 } = bbox
                ctx.strokeStyle = color
                ctx.lineWidth = lineWidth
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
            }
        }
    }

}

export function drawLandmarkOnCanvas(
    canvas: HTMLCanvasElement,
    imageURL: string,
    landmark: number[] | undefined,
    radius: number = 3
): void {
    const img = new Image();
    img.src = imageURL;

    img.onload = () => {
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        canvas.width = img.width;
        canvas.height = img.height;

        ctx.drawImage(img, 0, 0);

        ctx.fillStyle = "red";       
        ctx.strokeStyle = "white";   
        ctx.lineWidth = 1;
        
        if(landmark != undefined){
            for (let i = 0; i < landmark.length; i += 2) {
                const x = landmark[i];
                const y = landmark[i + 1];

                if (x === undefined || y === undefined) break;

                ctx.beginPath();
                ctx.arc(x, y, radius, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }
        }
    };
}
