const express= require('express');
const tf=require('@tensorflow/tfjs-node');
const bodyParser=require('body-parser');
const cv=require('./opencv');
const upload= require('express-fileupload');
const cocoSsd = require('@tensorflow-models/coco-ssd');
fs=require('fs');
const app=express();
app.use(upload());
inputSize = [300, 300];
mean = [127.5, 127.5, 127.5];
std = 0.007843;
swapRB = false;
confThreshold = 0.5;
nmsThreshold = 0.4;
outType = "SSD";
app.use(express.static("public"));
modelUrl="https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1";

const box=(bbox,img,text)=>{
    let left = bbox[0];
        let top = bbox[1];
        let width = bbox[2];
        let height = bbox[3];

        cv.rectangle(img, new cv.Point(left, top), new cv.Point(left + width, top + height),
                             new cv.Scalar(0, 255, 0));
        cv.rectangle(img, new cv.Point(left, top), new cv.Point(left + width, top + 15),
                             new cv.Scalar(255, 255, 255), cv.FILLED);
        cv.putText(img, text, new cv.Point(left, top + 10), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                                 new cv.Scalar(0, 0, 0));
}

main = async function(img,gray) {
    const model= tf.loadLayersModel('file://models/model.json');
    const model3= await cocoSsd.load({base:'mobilenet_v2'});
    predictions =  await model3.detect(img);
    for(let i=0;i<predictions.length;i++) {
        if(predictions[i].class=="motorcycle" || predictions[i].class=="person"){
            let image=tf.image.resizeBilinear(gray,[50,50])
            let im=image.expandDims(0);
            const pred= (await model).predict(im);
            console.log(pred.print());
            /*if(pred==0){
                box(predictions.bbox,img,'Helmet Not Found');
                console.log("Not Found")
            }else{
                box(predictions.bbox,img,'Helmet Found');
                console.log("Found")
            }*/
        }

    }
    
    console.log(predictions);
}

app.get('/',(req,res)=>{
    res.sendFile(__dirname+"/index.html");
})

app.get('/output',(req,res)=>{
    res.sendFile(__dirname+"/video.html")
})

app.post('/',async(req,res)=>{
    if(req.files){
        let file=req.files.file;
        let name=file.name;
        console.log(file.data);
        const img=tf.node.decodeImage(file.data);
        const rank1=tf.node.decodeImage(file.data,1);
        var tensor = tf.image.resizeBilinear(img, [320,320]);
        var tensor1 = tf.image.resizeBilinear(rank1, [320,320]);
        tensor=tensor.cast('int32');
        tensor1=tensor1.cast('int32');
        console.log(tensor1);
        main(tensor,tensor1)
    }
    res.redirect('/output')
})

app.listen(3000,()=>{
    console.log('listening on port 3000')
})