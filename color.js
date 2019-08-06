 
let model;

let resolution = 20;
let cols;
let rows;
let xs;

const train_xs = tf.tensor2d([
	[0.62,0.72,0.88], 
	[0.1,0.84,0.72], 
	[0.33,0.24,0.29], 
	[0.74,0.78,0.86],
	[0.31,0.35,0.41],
	[1,0.99,0],
	[1,0.42,0.52]
]);

const train_ys = tf.tensor2d([
	[1],
	[1],
	[0],
	[1],
	[0],
	[1],
	[0]
]);
 
 function setup()
 {
 	createCanvas(400,400);

 // 	cols = width / resolution;
	// rows = height / resolution;

	let inputs = [];
 	for(let i=0; i < 20; i++)
 	{
 		for(let j=0; j < 20; j++)
 		{
 			for(let k=0; k < 20; k++)
 			{
 				let x1 = i/20;
	 			let x2 = j/20;
	 			let x3 = k/20;
	 			inputs.push([x1, x2, x3]);
 			}
		}	
 	}
 	xs = tf.tensor2d(inputs);

 	model = tf.sequential();
 	let hidden = tf.layers.dense({
 		inputShape: [3],
 		units: 3,
 		activation: 'sigmoid'
 	});

 	let output = tf.layers.dense({
 		units: 1,
 		activation: 'sigmoid'
 	});

 	model.add(hidden);
 	model.add(output);

 	const optimizer = tf.train.adam(0.1);
 	model.compile({
 		optimizer: optimizer,
 		loss: 'meanSquaredError'
 	})

 	setTimeout(train, 100);
 }

function train() {
	 trainModel().then(result => {
	 console.log(result.history.loss[0]);
	 setTimeout(train, 100);
	 });
}

function trainModel(){
	return model.fit(train_xs, train_ys, {
		shuffle: true,
		epochs: 10
	});
}

 function draw()
 {
 	background(0);

 	tf.tidy(() => {

 		let ys = model.predict(xs);
		let y_values = ys.dataSync();
	// // console.log(ys);

		let index = 0;
		for(let i=0; i < 20; i++)
	 	{
	 		for(let j=0; j < 20; j++)
	 		{
	 			let br = y_values[index] * 255;
	 			fill(br);
	 			rect(i * resolution, j * resolution, resolution, resolution);
	 			fill(255 - br);
	 			textAlign(CENTER, CENTER);
	 			text(nf(y_values[index], 1, 2), i * resolution + resolution / 2, j * resolution + resolution / 2);
				
				index++;
	 		}	
	 	}
 });
	
 }