var points = [];

function main(){
    var canvas = document.getElementById('canvas1');
    if(!canvas || !canvas.getContext){
        console.log('error : can not load canvas');
        return false;
    }
    var context = canvas.getContext('2d');

    var img = new Image();
    img.src = "883957.png";
    img.onload = function(){
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
    }

    /* イベントリスナーの登録 */
    canvas.addEventListener(
        'click',
        function (e){
            /* get mouse position */
            var rect = e.target.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;

            /* plot cross dot */
            (function (x, y) {
                var size = 10;
                color = 'rgb(0, 255, 255)';
                context.beginPath();
                context.strokeStyle = color;
                context.moveTo(x-size/2, y-size/2);
                context.lineTo(x+size/2, y+size/2);
                context.stroke();
                context.moveTo(x-size/2, y+size/2);
                context.lineTo(x+size/2, y-size/2);
                context.stroke();
            })(x, y);

            p = [x, y*(-1) + canvas.height]
            points.push(p);
            console.log(p);
        },
        false
    );
}

function display() {
    target = document.getElementById("output1");
    var s = "";
    points.forEach(function(p){
        console.log(p);
        x = p[0];
        y = p[1];
        s += String(x) + ',' + String(y) + '<br />';
    });
    target.innerHTML = s;
}
