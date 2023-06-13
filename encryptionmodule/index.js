const ort = require('onnxruntime-node')
const nj = require('numjs');
const { BN } = require('bn.js');
const { randomBytes } = require('crypto');
const { float32 } = require('numjs/src/dtypes');
const ndarray = require("ndarray")
ndarray.ops = require("ndarray-ops")
const main = async () =>
{
const session = await ort.InferenceSession.create('arcface_final.onnx');
let pathali = "./pics/Ali_Naimi_";
let pathabd = "./pics/Abdullah_Gul_";

// not used for train/validation.
let true1 = nj.images.read(pathali+"0001.jpg");
let true2 = nj.images.read(pathali+"0002.jpg");
let true3 = nj.images.read(pathali+"0003.jpg");
let true4 = nj.images.read(pathali+"0004.jpg");
let true5 = nj.images.read(pathali+"0005.jpg");
let test1 = nj.images.read(pathali+"0006.jpg");
let false1 = nj.images.read(pathabd+"0001.jpg");

function preprocess(imarr)
{
    let image = ndarray(new Float32Array(imarr), [250, 250, 3])
    let newim = ndarray(new Float32Array(250 * 250 * 3), [1, 3, 250, 250])
    ndarray.ops.divseq(image, 255.0);
   
    ndarray.ops.assign(newim.pick(0, 0, null, null), image.pick(null, null, 2))
    ndarray.ops.assign(newim.pick(0, 1, null, null), image.pick(null, null, 1))
    ndarray.ops.assign(newim.pick(0, 2, null, null), image.pick(null, null, 0))

    return newim.data;
}
const datas = [true1, true2, true3, true4, true5, test1, false1];
const tensors = [];
for (let i = 0; i < 7; i++)
{
    let t = new ort.Tensor("float32", preprocess(datas[i].selection.data), [1, 3, 250, 250]);
    tensors.push(t);
}

const feats = []
for (let i = 0; i < 7; i++)
{
    let x = await session.run({input: tensors[i]});
    feats.push(x);
}

function init(module, len)
{

    module.reduction = BN.red('p192');
    let g = new BN(11);
    module.g = g.toRed(module.reduction);
    let pk = new BN(randomBytes(96));
    module.pk = pk;
    //console.log(module.g.red, module.pk.red)
    module.len = len;
    module.b = module.g.redPow(pk);
    //console.log(module.reduction);
}
function encryptX(module, x, fixed)
{
    let bnx = [];
    for(var i = 0; i < module.len; i++)
    {
        bnx.push(new BN(x[i]).toRed(module.reduction));
    }
    let r_ = fixed ? fixed : randomBytes(96);
    //console.log(r_)
    let r = new BN(r_);
    let gr = module.g.redPow(r);
    let br = module.b.redPow(r);
    //console.log("Br", br);
    let encx = [];
    for (var i = 0; i < module.len; i++)
    {
        let temp = bnx[i].redMul(br);
        encx.push(temp);
    }
    return [gr, encx];
}

function encryptY(module, y, gr)
{
    let rb = gr.redPow(module.pk);
    let rbinv = rb.redInvm();
    //console.log("Rb", rb, "rb * rbinv", rb.redMul(rbinv));
    let bny = [];
    for (var i = 0; i < module.len; i++)
    {
        bny.push(new BN(y[i]).toRed(module.reduction));
    }
    let ency = [];
    for (var i = 0; i < module.len; i++)
    {
        let temp = bny[i].redMul(rbinv);
        ency.push(temp);
    }
    return ency;
}

function innerprod(module, encx, ency)
{
    let data = [];
    for(var i = 0; i < module.len; i++)
    {
        data.push(encx[i].redMul(ency[i]));
    }
    let sum = new BN(0);
    for(var i = 0; i < module.len; i++)
    {
        sum = sum.add(data[i]);
    }
    return sum;
}

function innerprod_real(x, y, l)
{
    let sum = 0;
    for (var i = 0; i < l; i++)
    {
        sum +=  x[i] * y[i];
    }
    return sum;
}


// test for 5-len;
//let x = [1, 21, 4, 4, 6]
//let y = [10, 39, 8, 7, 9]
//console.log("HELLO WORLD!")
//const femoudle = {};
//init(femoudle, 5);
//let [gr, encx] = encryptX(femoudle, x);
//let ency = encryptY(femoudle, y, gr);
//let a = innerprod(femoudle, encx, ency).fromRed();
//let a2 = innerprod_real(x, y, 5);
//console.log(a, a2);


// test for 512-len;
const ipfe512 = {}
init(ipfe512, 512);
// console.log(resnormal, res); 
console.log("----preparing done----");
console.log("origin - test");
console.log(innerprod_real(feats[0]['/_module/flat/Flatten_output_0'].data, feats[5]['/_module/flat/Flatten_output_0'].data, 512))
console.log(innerprod_real(feats[1]['/_module/flat/Flatten_output_0'].data, feats[5]['/_module/flat/Flatten_output_0'].data, 512))
console.log(innerprod_real(feats[2]['/_module/flat/Flatten_output_0'].data, feats[5]['/_module/flat/Flatten_output_0'].data, 512))
console.log(innerprod_real(feats[3]['/_module/flat/Flatten_output_0'].data, feats[5]['/_module/flat/Flatten_output_0'].data, 512))
console.log(innerprod_real(feats[4]['/_module/flat/Flatten_output_0'].data, feats[5]['/_module/flat/Flatten_output_0'].data, 512))
console.log("origin - false");
console.log(innerprod_real(feats[0]['/_module/flat/Flatten_output_0'].data, feats[6]['/_module/flat/Flatten_output_0'].data, 512))
console.log(innerprod_real(feats[1]['/_module/flat/Flatten_output_0'].data, feats[6]['/_module/flat/Flatten_output_0'].data, 512))
console.log(innerprod_real(feats[2]['/_module/flat/Flatten_output_0'].data, feats[6]['/_module/flat/Flatten_output_0'].data, 512))
console.log(innerprod_real(feats[3]['/_module/flat/Flatten_output_0'].data, feats[6]['/_module/flat/Flatten_output_0'].data, 512))
console.log(innerprod_real(feats[4]['/_module/flat/Flatten_output_0'].data, feats[6]['/_module/flat/Flatten_output_0'].data, 512))
console.log("test - false");
console.log(innerprod_real(feats[5]['/_module/flat/Flatten_output_0'].data, feats[6]['/_module/flat/Flatten_output_0'].data, 512))
let test_s = Array.from(feats[0]['/_module/flat/Flatten_output_0'].data, (i, _) => i * 2 ** 48);
let test_t = Array.from(feats[5]['/_module/flat/Flatten_output_0'].data, (i, _) => i * 2 ** 48);
let test_f = Array.from(feats[6]['/_module/flat/Flatten_output_0'].data, (i, _) => i * 2 ** 48);
let [grst, encxs] = encryptX(ipfe512, test_s);
let encyt = encryptY(ipfe512, test_t, grst);
let res = innerprod(ipfe512, encxs, encyt)
console.log("origin - test in ipfe:", res.fromRed().div(new BN(2 ** 48).imul(new BN(2 ** 48))).toString(10));
let [grsf, encxs_] = encryptX(ipfe512, test_s);
let encyf = encryptY(ipfe512, test_f, grsf);
let res2 = innerprod(ipfe512, encxs_, encyf);
console.log("origin - false in ipfe:",res2.fromRed().div(new BN(2 ** 48).imul(new BN(2 ** 48))).toString(10));

}

main();
