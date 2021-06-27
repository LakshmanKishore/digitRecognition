window.onload = function () {

   let canvas = document.querySelector(".canvas");
   let clearButton = document.querySelector(".clearButton");
   let predictButton = document.querySelector(".predictButton");
   let result = document.querySelector(".result");
   canvas.height = 200;
   canvas.width = 200;
   canvas.style.backgroundColor = "black";

   let c = canvas.getContext("2d");
   let boundings = canvas.getBoundingClientRect();

   function getXY(e) {
      let xcord = e.clientX == undefined ? e.touches[0].clientX : e.clientX;
      let ycord = e.clientY == undefined ? e.touches[0].clientY : e.clientY;
      return [xcord - boundings.left, ycord - boundings.top];
   }

   let x, y;

   function draw(e) {
      [x, y] = getXY(e);
      c.strokeStyle = "white";
      c.lineWidth = 20;
      c.lineTo(x, y);
      c.stroke();
   }

   function startCaptureCord(e) {
      // console.log("Start Capture");
      [x, y] = getXY(e);
      c.beginPath();
      c.moveTo(x, y);
      canvas.addEventListener("touchmove", draw);
      canvas.addEventListener("mousemove", draw);
   }

   function stopCaptureCord(e) {
      // console.log("Stop Capture");
      canvas.removeEventListener("touchmove", draw);
      canvas.removeEventListener("mousemove", draw);
   }

   canvas.addEventListener("touchstart", startCaptureCord);
   canvas.addEventListener("mousedown", startCaptureCord);
   document.addEventListener("touchend", stopCaptureCord);
   document.addEventListener("mouseup", stopCaptureCord);


   function clear() {
      c.closePath();
      c.clearRect(0, 0, canvas.width, canvas.height);
   }

   clearButton.addEventListener("click", clear);

   function preprocessCanvas(image) {
      let tensor = tf.browser.fromPixels(image)
         .resizeNearestNeighbor([28, 28])
         .mean(2)
         .toFloat()
         .reshape([1, 784]);
      return tensor;
   }

   async function predictWithData(data) {
      let model = undefined;
      model = await tf.loadLayersModel("models/model.json");
      console.log("model loaded");
      // console.log(model);
      let output = model.predict(data);
      let outputArray = output.dataSync();
      let res = outputArray.indexOf(Math.max.apply(null, outputArray));
      console.log(res);
      result.innerText = res;
   }


   function predict() {
      let digitDataArr = preprocessCanvas(canvas)
      let digitData = digitDataArr.dataSync();
      console.log(digitData.length);
      let digitDataArray = [];
      console.log(digitData.data);
      for (let i = 0; i < digitData.length; i++) {
         digitDataArray[i] = digitData[i];
      }
      let digitData2d = [];
      while (digitDataArray.length) digitData2d.push(digitDataArray.splice(0, 28));
      // console.log(digitDataArr.dataSync());
      let predictData = tf.tensor3d([digitData2d]);
      console.log(predictData.dataSync());
      predictWithData(predictData);
   }

   predictButton.addEventListener("click", predict);

}
