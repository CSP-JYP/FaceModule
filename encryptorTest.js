import { BN } from 'bn.js';
import {randomBytes} from 'crypto'


/**

Encryption module based on ElGamal;
server -> client send: y
client -> server send: enc(x), enc(y), only modulo param p;
enc(x) dot enc(y) = x dot y

g <- Zp(gen)
beta* <- Zp
b = g^beta
r <- Zp
xi = b^r * xi
yi = inv(b^r) * yi
xi * b^r * yi * inv(b^r) === xi * yi mod p

**/



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
function encryptX(module, x)
{
    let bnx = [];
    for(var i = 0; i < module.len; i++)
    {
        bnx.push(new BN(x[i]).toRed(module.reduction));
    }
    let r = new BN(randomBytes(96));
    let gr = module.g.redPow(r);
    let br = module.b.redPow(r);
    console.log("Br", br);
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
    console.log("Rb", rb, "rb * rbinv", rb.redMul(rbinv));
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
    return new BN(sum);
}


// test for 512-len;
const ipfe512 = {}
init(ipfe512, 512);
let newx = Array.from({length: 512}, (_, i) => i+1);
let newy = Array.from({length: 512}, (_, i) => 512-i);
let [grnew, encxnew] = encryptX(ipfe512, newx);
let encynew = encryptY(ipfe512, newy, grnew);
let resnormal = innerprod_real(newx, newy, 512);
let res = innerprod(ipfe512, encxnew, encynew);
console.log(resnormal, res); 
