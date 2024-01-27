// https://observablehq.com/@stwind/source-hans-serif-a-glyph-atlas@1492
import define1 from "./a0c168d0a2572b78@1031.js";

function _1(md){return(
md`# Source Hans Serif: A Glyph Atlas`
)}

function _2(md){return(
md`[Source Hans Serif](https://source.typekit.com/source-han-serif/) is the second Pan-CJK typeface family from Adobe Type, a unified typeface design to serve the 1.5 billion people in East Asia. It supports four different East Asian languages — Simplified Chinese, Traditional Chinese, Japanese, and Korean — and the 65,535 glyphs in each of its seven weights are designed to work together with a consistent design that emphasizes shared elements between the languages while honoring the diversity of each.

This notebook is an attempt to explore all the glyphs in the typeface. Evolving from my last experiment of [Fashion MNIST Exploration](https://observablehq.com/@stwind/exploring-fashion-mnist), and getting huge inspiration from Google Art's [Runway Palette](https://artsexperiments.withgoogle.com/runwaypalette) experiment, I applied a similar approach to embed the glyphs into 2D and 3D spaces.

<sub>*Required WebGL2 browsers.*</sub>

---`
)}

function _glyphIdx(d3,glyphSets,html)
{
  const min = 0, max = d3.sum(glyphSets, x => x.length) - 1;
  let value = 0;
  const number = html`<input type="number" min="${min}" max="${max}" value=${min} step="1" name="number" required style="width:100px;" placeholder=${min}...${max} />`;
  const range = html`<input type="range" min="${min}" max="${max}"  value=${min} step="1" name="range" style="margin-left: 6.5px;flex-grow:1;" />`;
  const random = html`<button style="margin-left: 6.5px;">random</button>`;
  const form = html`<form onsubmit="event.preventDefault();" style="font: 13px/1.2 var(--sans-serif);display:flex;align-items:center;min-height:25.5px;">
    <label style="width:150px;padding:5px 0 4px 0;">glyph</label>
    <div style="display:flex;width:100%;align-items:center;">
      ${number}
      ${range}
      ${random}
    </div>
  </form>`;
  
  number.addEventListener("input", e => {
    value = range.valueAsNumber = e.target.valueAsNumber;
  });
  range.addEventListener("input", e => {
    value = number.valueAsNumber = e.target.valueAsNumber;
  });
  random.addEventListener("click", e => {
    value = Math.round(min + Math.random() * (max - min));
    range.valueAsNumber = value;
    number.valueAsNumber = value;
    form.dispatchEvent(new CustomEvent("input"));
  });
  
  Object.defineProperty(form, "value", {
    get() { return value; },
    set(v) {
      value = v;
      number.value = value;
      range.valueAsNumber = value;
      form.dispatchEvent(new CustomEvent("input"));
    }
  });
  
  return form;
}


function _is3d(Inputs){return(
Inputs.toggle({label: "3D", value: true})
)}

function _gl(DOM,width,html,loadingProgress)
{
  const canvas = DOM.canvas(width, width / 1.618);
  const gl = canvas.value = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');

  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
  const bg = 0.1;
  gl.clearColor(bg, bg, bg, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  
  const root = html`<div style="position:relative">
    <div style="color:white;position:absolute;top:0;left:0;height:100%;width:100%;display:flex;justify-content:center;align-items:center;pointer-events:none;user-select:none;">${loadingProgress()}</div>

    ${canvas}
  </div>`;
  root.value = gl;
  
  return root;
}


function _6(md){return(
md`---
## Process

The whole process involve the following steps:

1. Train an autoencoder with [PyTorch](https://pytorch.org/) to compress the glyph images into a latent space of 128 dimensions.
2. Use [Minimum-Distortion Embedding](https://pymde.org/) to embed the latent vectors into 2D and 3D spaces.
3. Use [GeomLoss](https://www.kernel-operations.io/geomloss/) and [Lap](https://github.com/gatagat/lap) to assign 2D embedding to grid.
4. Render vector fonts in WebGL2 with the [Slug Algorithm](http://jcgt.org/published/0006/02/02/).
5. Compress assets for the web with [Draco](https://github.com/google/draco) and [Zstandard](https://facebook.github.io/zstd/).

`
)}

async function _7(md,html,FileAttachment){return(
md`### 1. Finding latent features with Autoencoder

The first step was to render the glyphs as 64x64 images, then train an autoencoder with [Perceptual Similarity](https://github.com/richzhang/PerceptualSimilarity) loss.

Here is the neural network architecture:

~~~
==============================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==============================================================================
Autoencoder                              --                        --
├─Encoder: 1-1                           [2, 128]                  --
│    └─Sequential: 2-1                   [2, 128, 1, 1]            --
│    │    └─Conv2d: 3-1                  [2, 32, 64, 64]           320
│    │    └─BatchNorm2d: 3-2             [2, 32, 64, 64]           64
│    │    └─ReLU: 3-3                    [2, 32, 64, 64]           --
│    │    └─EncoderBlock: 3-4            [2, 32, 32, 32]           53,728
│    │    └─EncoderBlock: 3-5            [2, 64, 16, 16]           70,208
│    │    └─EncoderBlock: 3-6            [2, 128, 8, 8]            279,680
│    │    └─EncoderBlock: 3-7            [2, 256, 4, 4]            1,116,416
│    │    └─EncoderBlock: 3-8            [2, 128, 1, 1]            2,887,040
├─Decoder: 1-2                           [2, 1, 64, 64]            --
│    └─Sequential: 2-2                   [2, 128]                  --
│    │    └─Linear: 3-9                  [2, 128]                  16,512
│    │    └─BatchNorm1d: 3-10            [2, 128]                  256
│    │    └─ReLU: 3-11                   [2, 128]                  --
│    └─Sequential: 2-3                   [2, 1, 64, 64]            --
│    │    └─DecoderBlock: 3-12           [2, 256, 4, 4]            2,887,424
│    │    └─DecoderBlock: 3-13           [2, 128, 8, 8]            1,116,032
│    │    └─DecoderBlock: 3-14           [2, 64, 16, 16]           279,488
│    │    └─DecoderBlock: 3-15           [2, 32, 32, 32]           70,112
│    │    └─DecoderBlock: 3-16           [2, 32, 64, 64]           53,728
│    │    └─Conv2d: 3-17                 [2, 1, 64, 64]            289
│    │    └─Sigmoid: 3-18                [2, 1, 64, 64]            --
==============================================================================
Total params: 8,831,297
Trainable params: 8,831,297
Non-trainable params: 0
Total mult-adds (G): 9.11
==============================================================================
~~~

The network was trained for 300 epochs, here are the stats of last 30 epochs

${html`<figure align="center">
  <img src="${await FileAttachment("fig_ae_loss.png").url()}" width="600" />
  <figcaption>Loss, PSNR and SSIM of last 30 epochs</figcaption>
</figure>`}

${html`<figure align="center">
  <img src="${await FileAttachment("fig_ae_res.png").url()}" width="600" />
  <figcaption>Comparison of original data and autoencoder reconstructions.</figcaption>
</figure>`}`
)}

async function _8(md,html,FileAttachment){return(
md`### 2. Generating 2D and 3D Embeddings.

Once we have the 128D latent features, we can use [Minimum-Distortion Embedding](https://pymde.org/) to generate the 2D and 3D embeddings. What makes MDE better than UMAP is that you have more controls of the overall shapes of the embeddings.

${html`<figure align="center">
  <img src="${await FileAttachment("fig_mde_2d.png").url()}" width="600" />
  <figcaption>MDE 2D</figcaption>
</figure>`}

${html`<figure align="center">
  <img src="${await FileAttachment("fig_mde_3d.png").url()}" width="600" />
  <figcaption>MDE 3D</figcaption>
</figure>`}

The we can use [Draco](https://github.com/google/draco) to compress the resulting point cloud.`
)}

async function _9(md,html,FileAttachment){return(
md`### 3. Assigning 2D Embedding to Grid

Next step is to assign the 2d embedding to a grid with [GeomLoss](https://www.kernel-operations.io/geomloss/), as the same with [Runway Pallete](https://artsexperiments.withgoogle.com/runwaypalette) by [Cyril Diagne](https://twitter.com/cyrildiagne/status/1199016369662152704). 

But GeomLoss is an approximation not a perfect assignment, so it leaves you about 5~8% of duplications and holes:

${html`<figure align="center">
  <img src="${await FileAttachment("fig_geomloss_grid.png").url()}" width="600" />
  <figcaption>Grid assignment with geomloss.</figcaption>
</figure>`}

For the 4,752 misplacements out of 64,033 assignments, we can use [Lap](https://github.com/gatagat/lap) to nudge them to the final locations.

${html`<figure align="center">
  <img src="${await FileAttachment("fig_geomloss_lap.png").url()}" width="600" />
  <figcaption>Lap assignment for duplications and holes.</figcaption>
</figure>`}`
)}

async function _10(md,html,FileAttachment){return(
md`### 4. Render WebGL2 Vector Fonts With Slug Algorithm

The [Slug Algorithm](http://jcgt.org/published/0006/02/02/) is the state of art GPU multi-resolutions vector font rendering technique.

The general idea is to split the em unit into 16 vectical and horizontal bands, and for each pixel we can just check the glyph curves intersecting the corresponding bands.

${html`<figure align="center">
  <img src="${await FileAttachment("fig_slug.png").url()}" width="600" />
  <figcaption>Splitting em unit into 16 vertical and horizontal bands</figcaption>
</figure>`}

### 5. Compressing Assets for the Web

The final challenge is that there are total 64,033 valid glyphs, which has 64MB curves data. To make it more efficient for the web, we can split them into 4 glyph sets, and compress each with [Zstandard](https://facebook.github.io/zstd/). I compiled a simple [WASM version of Zstandard](https://www.npmjs.com/package/zstd-wasm) with [Emscripten](https://emscripten.org/), which just takes 19kb transfer after GZIP. The compression of glyph sets at level 22 saves another 30% spaces, and we can download and decode them in parallel with Web Workers.
`
)}

function _11(md){return(
md`---
## Code`
)}

function _render(gl,picking,camera,t,atlases,hover,selected)
{
  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
  picking.draw(camera, t);
  
  const bg = 0.15;
  gl.clearColor(bg, bg, bg, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // axis.draw(camera);
  atlases.forEach(x => x.draw(camera, t, hover, selected));
}


function _t(atlases,approach,is3d)
{ atlases; return approach(this === undefined ? 1: this, is3d ? 0 : 1, 0.08, 1e-5) }


function _hover(Generators,d3,gl,picking){return(
Generators.observe(next => {
  const canvas = d3.select(gl.canvas);
  let cur = undefined;
  next(cur);
  
  canvas.on('mousemove', e => {
    const id = picking.pick(e.offsetX, gl.drawingBufferHeight - e.offsetY);
    if (id != cur) {
      next(cur = id);
    }
  });
  canvas.on("mouseout", e => {
    next(cur = undefined);
  });

  return () => canvas.on('mousemove', null).on("mouseout", null);
})
)}

function _selected(debounce,$0){return(
debounce($0, 300)
)}

function _16(atlases,d3,gl,picking,$0,invalidation)
{
  atlases;
  
  const canvas = d3.select(gl.canvas);
  canvas.on('click', e => {
    const id = picking.pick(e.offsetX, gl.drawingBufferHeight - e.offsetY);
    if (id) {
      $0.value = id;
    }
  });
  invalidation.then(() => canvas.on("click", null));
  return "initialized";
}


function _focus(atlases,$0,$1,v,is3d,embedding,selected)
{
  atlases;
  
  let target = [0, 0, 0];
  if ($0.value) {
    $1.goto({ eye: [0,0,7.5], target: v.ZERO3, up: v.Y3 }, 0.09, 1e-4);
  } else if (is3d) {
    target[0] = embedding.position[selected * 3];
    target[1] = embedding.position[selected * 3 + 1];
    target[2] = embedding.position[selected * 3 + 2];
    const eye = v.mulN(null, v.normalize([], target), v.mag(target) + 0.025);
    $1.goto({ eye, target: v.ZERO3 }, 0.1, 1e-4);
  } else {
    target[0] = embedding.grid[selected * 2];
    target[1] = embedding.grid[selected * 2 + 1];
    const eye = v.add3([], target, [0,0,0.25]);
    $1.goto({ eye, target, up: v.Y3 }, 0.1, 1e-4);
  }
}


function _18($0,d3,gl,is3d,$1,invalidation)
{
  const cam = $0;
  const canvas = d3.select(gl.canvas);
  if (is3d) {
    canvas.call(d3.drag().on('start', e => {
      $1.value = true;
      cam.orbitStart(e.x, e.y);
    }).on('drag', e => cam.orbitUpdate(e.x, e.y)))
      .on('wheel', e => {
        e.preventDefault();
        $1.value = true;
        cam.zoom(e.deltaY * 1e-3, 0.01, 200 - 1);
      });
  } else {
    canvas.call(d3.drag().on('start', e => {
      $1.value = true;
      cam.panStart(e.x, e.y);
    }).on('drag', e => cam.panUpdate(e.x, e.y)))
      .on('wheel', e => {
        e.preventDefault();
        $1.value = true;
        cam.panZoom(e.offsetX, e.offsetY, e.deltaY * 1e-3, 0.1, 200 - 1);
      });
  }
  
  invalidation.then(() => canvas.on('.drag', null).on('wheel', null));
  return "camera control";
}


function _freeView($0,glyphIdx)
{
  $0.value = glyphIdx == 0;
}


function _free(View){return(
new View(true)
)}

function _21(md){return(
md`### Embedding`
)}

async function _embedding(FileAttachment,dracoDecode,draco,makeGrid,d3)
{
  const data = await FileAttachment("embedding.drc").arrayBuffer();
  const raw = dracoDecode(data, {
    position: draco.DT_FLOAT32,
    grid: draco.DT_UINT16,
    index: draco.DT_UINT16
  });
  
  const index = raw.index, n = raw.index.length;
  const position = new Float32Array(n * 3), grid = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const idx = index[i];
    for (let k = 0; k < 3; k++)
      position[idx * 3 + k] = raw.position[i * 3 + k];
    for (let k = 0; k < 2; k++)
      grid[idx * 2 + k] = raw.grid[i * 2 + k];
  }
  return { 
    position, 
    grid: makeGrid(position, grid, index), 
    index: new Float32Array(d3.sort(index)) 
  };
}


function _embeddings(subEmbedding,embedding){return(
[
  subEmbedding(embedding, 0, 16384),
  subEmbedding(embedding, 16384, 32768),
  subEmbedding(embedding, 32768, 49152),
  subEmbedding(embedding, 49152, 64033),
]
)}

function _subEmbedding(){return(
(embedding, start, end) => ({
  position: embedding.position.slice(start * 3, end * 3),
  grid: embedding.grid.slice(start * 2, end * 3),
  index: embedding.index.slice(start, end)
})
)}

function _makeGrid(d3){return(
(position, grid, index) => {
  const domainX = d3.extent(index, i => grid[i * 2]);
  const domainY = d3.extent(index, i => grid[i * 2 + 1]);
  const extentX = d3.extent(index, i => position[i * 3]);
  const extentY = d3.extent(index, i => position[i * 3 + 1]);
  const extent = [Math.min(extentX[0], extentY[0]), Math.max(extentX[1], extentY[1])];
  const aspect = domainX[1] / domainY[1];
 
  const scale = {
    x: d3.scaleLinear(domainX, [extent[0] * aspect, extent[1] * aspect]),
    y: d3.scaleLinear([domainY[1], domainY[0]], extent),
  };
  const res = [], n = index.length;
  for (let i = 0; i < n; i++) {
    const x = grid[i * 2], y = grid[i * 2 + 1];
    res.push(scale.x(x), scale.y(y));
  }
  return new Float32Array(res);
}
)}

function _26(md){return(
md`### Slug`
)}

function _slugReady(atlases,hub)
{
  atlases;
  hub.publish("slugs", true);
}


async function _font0(buildSlug,FileAttachment,glyphSets,Font)
{
  const slug = await buildSlug("part01", await FileAttachment("SourceHanSerifTC-Bold_data_0.bin").url(), glyphSets[0]);
  return new Font(slug, { curveCoordsTexWidth: 2048, bandsTexWidth: 1024 });
}


async function _font1(buildSlug,FileAttachment,glyphSets,Font)
{
  const slug = await buildSlug("part02", await FileAttachment("SourceHanSerifTC-Bold_data_1.bin").url(), glyphSets[1]);
  return new Font(slug, { curveCoordsTexWidth: 2048, bandsTexWidth: 1024 });
}


async function _font2(buildSlug,FileAttachment,glyphSets,Font)
{
  const slug = await buildSlug("part03", await FileAttachment("SourceHanSerifTC-Bold_data_2.bin").url(), glyphSets[2]);
  return new Font(slug, { curveCoordsTexWidth: 2048, bandsTexWidth: 1024 });
}


async function _font3(buildSlug,FileAttachment,glyphSets,Font)
{
  const slug = await buildSlug("part04", await FileAttachment("SourceHanSerifTC-Bold_data_3.bin").url(), glyphSets[3]);
  return new Font(slug, { curveCoordsTexWidth: 2048, bandsTexWidth: 1024 });
}


function _picking(atlases,Picking,embedding)
{
  atlases;
  return new Picking(embedding);
}


function _atlases(Atlas,embeddings,font0,font1,font2,font3){return(
[
  new Atlas(embeddings[0], font0),
  new Atlas(embeddings[1], font1),
  new Atlas(embeddings[2], font2),
  new Atlas(embeddings[3], font3),
]
)}

async function _glyphSets(FileAttachment,decompress)
{
  const raw = new Uint8Array(await FileAttachment("SourceHanSerifTC-Bold_glyphs.json@1.zst").arrayBuffer());
  const data = decompress(raw);
  const text = new TextDecoder().decode(data);
  return JSON.parse(text);
}


function _buildSlug(fetchProgress,reportProgress,makeSlug,decodeCurvesData){return(
async (name, url, glyphs) => {
  const data = new Uint8Array(await fetchProgress(url, null, reportProgress(`${name}.download`)));
  return makeSlug(decodeCurvesData(data), glyphs, 4096, 16, reportProgress(`${name}.build`));
}
)}

function _makeSlug(rs)
{
  const source = `
const getCurves = (data, glyph) => {
  const { offset, num_curves } = glyph;
  const res = [];
  let j = offset;
  for (let i = 0; i < num_curves; i++) {
    let p1 = [data[j + 0], data[j + 1]];
    let p2 = [data[j + 2], data[j + 3]];
    let p3 = [data[j + 4], data[j + 5]];
    let first = j == offset;

    if (p2[0] == 1 && p2[1] == 1) {
      first = true;
      j += 4;
      p1 = [data[j + 0], data[j + 1]];
      p2 = [data[j + 2], data[j + 3]];
      p3 = [data[j + 4], data[j + 5]];
    }
    res.push({ p1, p2, p3, first });
    j += 4;
  }
  return res;
};

const maxP = (c, i) => Math.max(c.p1[i], c.p2[i], c.p3[i]);
const minP = (c, i) => Math.min(c.p1[i], c.p2[i], c.p3[i]);
const compareCurve = (curves, k) => (i, j) => {
  const xi = maxP(curves[i], k), xj = maxP(curves[j], k);
  if (xi == xj) return i < j ? -1 : 1;
  else return xi < xj ? 1 : -1;
};

const range = n => Array.from({ length: n }, (_, i) => i);

let curvesData = [], curveCoords = [], bands = [];
let data, texWidth, numBands;

self.addEventListener("message", e => {
  const { type } = e.data;
  if (type == "init") {
    data = e.data.data;
    texWidth = e.data.texWidth;
    numBands = e.data.numBands;

    self.postMessage({ type: "init" });
  } else if (type == "update") {
    const glyph = e.data.glyph;
    
    const curves = getCurves(data, glyph);
    const coords = [];
    for (const { p1, p2, p3, first } of curves) {
      if (first && curvesData.length % 4 != 0) {
        const toAdd = 4 - curvesData.length % 4;
        for (let i = 0; i < toAdd; i++) curvesData.push(1);
       }
      const newRow = Math.floor(curvesData.length / 4) % texWidth == texWidth - 1;
      if (newRow) {
        const toAdd = 8 - curvesData.length % 4;
        for (let i = 0; i < toAdd; i++) curvesData.push(1);
      }

      const n = Math.floor(curvesData.length / 4);
      coords.push([n % texWidth, Math.floor(n / texWidth)]);

      if (first || newRow) curvesData.push(...p1);
      curvesData.push(...p2, ...p3);
    }

    const indicesRL = range(curves.length).sort(compareCurve(curves, 0));
    for (let b = 0; b < numBands; b++) {
      const minY = b / numBands, maxY = (b + 1) / numBands, offset = curveCoords.length;
      let cnt = 0
      for (const i of indicesRL) {
        const mnY = minP(curves[i], 1), mxY = maxP(curves[i], 1);
        if (mnY > maxY || mxY < minY || mxY - mnY < 1e-5) continue;
        cnt++;
        curveCoords.push(coords[i]);
      }
      bands.push([cnt, offset]);
    }

    const indicesTB = range(curves.length).sort(compareCurve(curves, 1));
    for (let b = 0; b < numBands; b++) {
      const minX = b / numBands, maxX = (b + 1) / numBands, offset = curveCoords.length;
      let cnt = 0
      for (const i of indicesTB) {
        const mnX = minP(curves[i], 0), mxX = maxP(curves[i], 0);
        if (mnX > maxX || mxX < minX || mxX - mnX < 1e-5) continue;
        cnt++;
        curveCoords.push(coords[i]);
      }
      bands.push([cnt, offset]);
    }

    self.postMessage({ type: "update" });
  } else if (type == "done") {
    curvesData = new Float32Array(curvesData.flat(1));
    curveCoordsData = new Float32Array(curveCoords.flat(1));
    bandsData = new Float32Array(bands.flat(1));

    self.postMessage({ type: "done", curvesData, curveCoordsData, bandsData }, 
                     [curvesData.buffer, curveCoordsData.buffer, bandsData.buffer]);
  }
});
`;
  
  return (data, glyphs, texWidth = 4096, numBands = 16, progress = null) => {
    return new Promise(resolve => {
      const worker = rs.inlineWorker(source);
      let i = 0, n = glyphs.length;
      
      worker.addEventListener("message", e => {
        const { type } = e.data;
        if (type == "init") {
          worker.postMessage({ type: "update", glyph: glyphs[i++] });
        } else if (type == "update") {
          if (i == n) {
            worker.postMessage({ type: "done" });
          } else {
            worker.postMessage({ type: "update", glyph: glyphs[i++] });
            progress && progress(i, n);
          }
        } else if (type == "done") {
          worker.terminate();
          const { curvesData, curveCoordsData, bandsData } = e.data;
          resolve({ curvesData, curveCoordsData, bandsData, texWidth, numBands, glyphs });
        }
      });
      
      worker.postMessage({ type: "init", texWidth, numBands, data }, [data.buffer]);
    });
  };
}


function _Picking(webgl,gl,d3,ID,m){return(
class Picking {
  constructor(embedding) {
    this.model = webgl.compileModel(gl, {
      attribs: {
        position: { data: embedding.position, size: 3 },
        grid: { data: embedding.grid, size: 2 },
        index: { data: new Float32Array(d3.range(embedding.index.length)), size: 1 },
        id: { data: new Float32Array(Array.from(embedding.index, ID.encode).flat(1)) }
      },
      num: embedding.position.length / 3, mode: gl.POINTS, uniforms: {},
      shader: webgl.defShader(gl, {
        vs: `
void main() {
  v_index = index;
  v_id = id;

  vec3 p = mix(position, vec3(grid, 0.0), t);
  vec4 pos = view * model * vec4(p, 1.0);
  gl_Position = proj * pos;
  gl_PointSize = max(pointSize / -pos.z, 1.0);
}`,
        fs: `
void main() {
  fragColor = vec4(v_id, 0.0);
}`,
        declPrefixes: { v: "v_" },
        attribs: {
          position: "vec3",
          grid: "vec2",
          index: "float",
          id: "vec3"
        },
        varying: {
          index: "float",
          id: "vec3",
        },
        uniforms: {
          model: ["mat4", m.identity44([])],
          view: "mat4",
          proj: "mat4",
          
          pointSize: ['float', 20.6],
          t: ["float", 0],
        },
        state: {
          depth: true,
          cull: true,
          blend: false,
          blendFn: webgl.BLEND_NORMAL
        }
      })
    });
    this.fbo = webgl.defFBO(gl, { 
      tex: [
        webgl.defTexture(gl, { 
          width: gl.drawingBufferWidth, 
          height: gl.drawingBufferHeight, 
          premultiply: true, 
          image: null 
        }),
      ],
      depth: webgl.defRBO(gl, { 
        width: gl.drawingBufferWidth, 
        height: gl.drawingBufferHeight 
      })
    });
    Object.defineProperties(this, {
      _buf: { value: new Uint8Array(4) },
    });
  }
  
  pick(x, y) {
    this.fbo.bind();
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, this._buf);
    this.fbo.unbind();

    const id = ID.decode(this._buf);
    return id != 0xffffff ? id : undefined;
  }

  draw(camera, t) {
    this.fbo.bind();
    gl.clearColor(1.0, 1.0, 1.0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    this.model.uniforms.view = camera.view;
    this.model.uniforms.proj = camera.proj;

    this.model.uniforms.t = t;
    webgl.draw(this.model);
    this.fbo.unbind();
  }
}
)}

function _Atlas(webgl,gl,d3,ID,moddiv,tangentAt,gradAt,locAt,getRootCode,traceRay,getCurveCoord,traceRayBandH,traceRayBandV,aastep,frame,norm,m){return(
class Atlas {
  constructor(embedding, font) {
    this.model = webgl.compileModel(gl, {
      attribs: {
        position: { data: embedding.position, size: 3 },
        grid: { data: embedding.grid, size: 2 },
        index: { data: new Float32Array(d3.range(font.glyphs.length)), size: 1 },
        id: { data: new Float32Array(Array.from(embedding.index, ID.encode).flat(1)) }
      },
      num: embedding.position.length / 3, mode: gl.POINTS, uniforms: {},
      shader: webgl.defShader(gl, {
        vs: `
void main() {
  v_index = index;
  v_id = id;

  vec3 p = mix(position, vec3(grid, 0.0), t);
  vec4 pos = view * model * vec4(p, 1.0);
  gl_Position = proj * pos;
  v_size = max(pointSize / -pos.z, 1.0);
  gl_PointSize = v_size;
}`,
        fs: `
${moddiv}
${tangentAt}
${gradAt}
${locAt}
${getRootCode}
${traceRay}
${getCurveCoord}
${traceRayBandH}
${traceRayBandV}
${aastep}
${frame}
${norm}

void main() {
  vec2 uv = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y);
  vec2 pixelsPerEm = 1.0 / fwidth(uv);

  float bandIdx = v_index * bandsCount * 2.0;
  ivec2 bandCoord = ivec2(moddiv(bandIdx, bandsTexWidth));
  vec2 bandOffset = vec2(uv * bandsCount);

  float t0 = clamp(norm(v_size, 1.0, 7.0),0.0,1.0);
  vec2 hband = texelFetch(bandsTex, bandCoord + ivec2(bandOffset.y, 0), 0).xy;
  hband.x *= t0;
  vec2 resH = traceRayBandH(ivec2(hband), pixelsPerEm.x, uv);

  vec2 vband = texelFetch(bandsTex, bandCoord + ivec2(bandsCount + bandOffset.x, 0), 0).xy;
  vband.x *= t0;
  vec2 resV = traceRayBandV(ivec2(vband), pixelsPerEm.y, uv);

  float r = resV.y / (resH.y + resV.y + 1e-6);
  float alpha = mix(resH.x, resV.x, r);

  float col = mix(1.0, 0.15, alpha);
  float t = clamp(norm(v_size, 1.0, 100.0),0.0,1.0);
  float ft = frame(uv, t);

  col = mix(0.2, col, ft);
  if (selected != vec3(0.0) && distance(selected, v_id) < 1e-3)
    col = mix(1.0, 0.0, 1.0 - alpha);
  if (hover != vec3(0.0) && distance(hover, v_id) < 1e-3)
    col = mix(1.0, 0.2, 1.0 - alpha);

  fragColor = vec4(vec3(col), 1.0);
}`,
        declPrefixes: { v: "v_" },
        attribs: {
          position: "vec3",
          grid: "vec2",
          index: "float",
          id: "vec3"
        },
        varying: {
          index: "float",
          size: "float",
          id: "vec3"
        },
        uniforms: {
          model: ["mat4", m.identity44([])],
          view: "mat4",
          proj: "mat4",
          
          pointSize: ['float', 20.6],
          t: ["float", 0],
          hover: 'vec3',
          selected: 'vec3',
          
          bandsCount: ["float", font.numBands],
          bandsTexWidth: ["float", font.bandsTexWidth],
          curveCoordsTexWidth: ["float", font.curveCoordsTexWidth],
          
          bandsTex: ["sampler2D", 0],
          curveOffsetTex: ["sampler2D", 1],
          curvesTex: ["sampler2D", 2],
        },
        state: {
          depth: true,
          cull: true,
          blend: false,
          blendFn: webgl.BLEND_NORMAL
        }
      }),
      textures: [font.textures.bandsTex, font.textures.curveCoordsTex, font.textures.curvesTex]
    });
  }
  
  draw(camera, t, hover, selected) {
    this.model.uniforms.t = t;
    
    this.model.uniforms.view = camera.view;
    this.model.uniforms.proj = camera.proj;
    this.model.uniforms.hover = ID.encode(hover);
    this.model.uniforms.selected = ID.encode(selected);
    webgl.draw(this.model);
  }
}
)}

function _Font(webgl,gl)
{
  const padTo = (source, width, depth = 4, val = 0) => {
    const m = source.length;
    const line = width * depth
    const n = (line - source.length % line) % line;

    const ret = new source.constructor(m + n);
    ret.fill(val);
    ret.set(source);

    return ret;
  };
  
  return class {
    constructor(slug, { curveCoordsTexWidth, bandsTexWidth }) {
      const curvesData = padTo(slug.curvesData, slug.texWidth, 4);
      const curveCoordsData = padTo(slug.curveCoordsData, curveCoordsTexWidth, 2);
      const bandsData = padTo(slug.bandsData, bandsTexWidth, 2);

      this.textures = {
        curvesTex: webgl.defTextureFloat(gl, curvesData, slug.texWidth, 
                                         curvesData.length / slug.texWidth / 4),
        curveCoordsTex: webgl.defTextureFloat(gl, new Float32Array(curveCoordsData), curveCoordsTexWidth,
                                              Math.ceil(curveCoordsData.length / curveCoordsTexWidth / 2),
                                              webgl.TextureFormat.RG32F),
        bandsTex: webgl.defTextureFloat(gl, new Float32Array(bandsData), bandsTexWidth, 
                                        Math.ceil(bandsData.length / bandsTexWidth / 2),
                                        webgl.TextureFormat.RG32F),
      };
      this.curveCoordsTexWidth = curveCoordsTexWidth;
      this.bandsTexWidth = bandsTexWidth;
      this.numBands = slug.numBands;
      this.glyphs = slug.glyphs;
    }
  };
}


function _decodeCurvesData(decompress)
{
  const dequantize = (xQ, alpha, beta, alphaQ = 0, betaQ = 65535) => {
    const s = (beta - alpha) / (betaQ - alphaQ)
    const z = Math.round((beta * alphaQ - alpha * betaQ) / (beta - alpha));
    const res = new Float32Array(xQ.length);
    xQ.forEach((v, i) => res[i] = s * (v - z));
    return res;
  };
  
  const uint8ToUint16 = data => {
    const n = data.length / 2;
    const res = new Uint16Array(n);
    for (let i = 0; i < n; i ++) {
      res[i] = data[i * 2] + data[i * 2 + 1] * 256;
    }
    return res;
  };
  
  return data => dequantize(uint8ToUint16(decompress(data)), 0, 1, 0, 65535);
}


function _getRootCode(glsl){return(
glsl`
uint getRootCode(vec2 p1, vec2 p2, vec2 p3) {
  uint shift = ((p1.y > 0.0) ? 2U : 0U) + ((p2.y > 0.0) ? 4U : 0U) + ((p3.y > 0.0) ? 8U : 0U);
  return 0x2E74U >> shift & 3U;
}`
)}

function _traceRay(glsl){return(
glsl`
vec3 traceRay(vec2 p1, vec2 p2, vec2 p3, float pixelsPerEm) {
  uint code = getRootCode(p1, p2, p3);

  float coverage = 0.0;
  vec2 t = vec2(-1.0);

  if (code == 0U) 
    return vec3(coverage, t);
  
  vec2 a = p1 - p2 * 2.0f + p3, b = p1 - p2;
  float c = p1.y;
  float d = sqrt(max(b.y * b.y - a.y * c, 0.0));
  
  if (abs(a.y) < 1e-4) 
    t = vec2(c / (2.0 * b.y));
  else
    t = vec2(b.y - d, b.y + d) / a.y;

  vec2 x = (a.x * t - b.x * 2.0) * t + p1.x;
  vec2 cov = clamp(x * pixelsPerEm + 0.5, 0.0, 1.0);

  coverage += cov.x * float(code & 1U);
  coverage -= cov.y * float(code >> 1U);
  
  return vec3(coverage, t);
}`
)}

function _getCurveCoord(glsl){return(
glsl`
ivec2 getCurveCoord(sampler2D tex, float offset, float width) {
  ivec2 coord = ivec2(moddiv(offset, width));
  return ivec2(texelFetch(tex, coord, 0).xy);
}`
)}

function _traceRayBandH(glsl){return(
glsl`
vec2 traceRayBandH(ivec2 band, float pixelsPerEm, vec2 p) {
  float cov = 0.0, deriv = 0.0, closest = 100.0;

  for (int i = 0; i < band.x; i++) {
    ivec2 coord = getCurveCoord(curveOffsetTex, float(band.y + i), curveCoordsTexWidth);

    vec4 p12 = texelFetch(curvesTex, coord, 0) - vec4(p, p);
    vec2 p3 = texelFetch(curvesTex, coord + ivec2(1, 0), 0).xy - p;

    if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm < -0.5) break;

    vec3 res = traceRay(p12.xy, p12.zw, p3.xy, pixelsPerEm);
    cov += res.x;

    vec2 t = res.yz;
    vec2 loc = abs(locAt(p12.x, p12.z, p3.x, t));
    if (t.x >= 0.0 && loc.x < closest) {
      closest = loc.x;
      deriv = abs(gradAt(p12.xy, p12.zw, p3.xy, t.x));
    }
    if (t.y >= 0.0 && loc.y < closest) {
      closest = loc.y;
      deriv = abs(gradAt(p12.xy, p12.zw, p3.xy, t.y));
    }
  }
  return vec2(clamp(abs(cov), 0.0, 1.0), deriv);
}`
)}

function _traceRayBandV(glsl){return(
glsl`
vec2 traceRayBandV(ivec2 band, float pixelsPerEm, vec2 p) {
  float cov = 0.0, deriv = 0.0, closest = 100.0;

  for (int i = 0; i < band.x; i++) {
    ivec2 coord = getCurveCoord(curveOffsetTex, float(band.y + i), curveCoordsTexWidth);

    vec4 p12 = (texelFetch(curvesTex, coord, 0) - vec4(p, p)).yxwz;
    vec2 p3 = (texelFetch(curvesTex, coord + ivec2(1, 0), 0).xy - p).yx;

    if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm < -0.5) break;

    vec3 res = traceRay(p12.xy, p12.zw, p3.xy, pixelsPerEm);
    cov += res.x;

    vec2 t = res.yz;
    vec2 loc = abs(locAt(p12.x, p12.z, p3.x, t));
    if (t.x >= 0.0 && loc.x < closest) {
      closest = loc.x;
      deriv = abs(gradAt(p12.xy, p12.zw, p3.xy, t.x));
    }
    if (t.y >= 0.0 && loc.y < closest) {
      closest = loc.y;
      deriv = abs(gradAt(p12.xy, p12.zw, p3.xy, t.y));
    }
  }
  return vec2(clamp(abs(cov), 0.0, 1.0), deriv);
}`
)}

function _moddiv(glsl){return(
glsl`
vec2 moddiv(float a, float b) {
  return vec2(mod(a, b), a / b);
}`
)}

function _pointAt(glsl){return(
glsl`
vec2 pointAt(vec2 p1, vec2 p2, vec2 p3, float t) {
  float u = 1.0 - t;
  return u * u * p1 + 2.0 * t * u * p2 + t * t * p3;
}`
)}

function _tangentAt(glsl){return(
glsl`
float tangentAt(float p0, float p1, float p2, float t) {
	return 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1);
}

vec2 tangentAt(vec2 p0, vec2 p1, vec2 p2, float t) {
	return 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1);
}`
)}

function _gradAt(glsl){return(
glsl`
float gradAt(vec2 p1, vec2 p2, vec2 p3, float t) {
  vec2 d = tangentAt(p1, p2, p3, t);
  return d.y / (d.x + 1e-6);
}`
)}

function _locAt(glsl){return(
glsl`
vec2 locAt(float x1, float x2, float x3, vec2 t) {
  vec2 u = 1.0 - t;
  return u * u * x1 + 2.0 * t * u * x2 + t * t * x3;
}`
)}

function _aastep(glsl){return(
glsl`
float aastep(float threshold, float value) {
  float afwidth = length(vec2(dFdx(value), dFdy(value))) * 0.70710678118654757;
  return smoothstep(threshold-afwidth, threshold+afwidth, value);
}

vec2 aastep(vec2 threshold, vec2 value) {
  vec2 dx = dFdx(value), dy = dFdy(value);
  float afw1 = length(vec2(dx.x, dy.x)), afw2 = length(vec2(dx.y, dy.y));
  vec2 afwidth = vec2(afw1, afw2) * 0.70710678118654757;
  return smoothstep(threshold-afwidth, threshold+afwidth, value);
}`
)}

function _frame(glsl){return(
glsl`
float frame(vec2 uv, float width) {
  vec2 d = fwidth(uv) * width;
  vec2 t = aastep(d, uv) * (1.0 - aastep(1.0 - d, uv));
  return t.x * t.y;
}`
)}

function _norm(glsl){return(
glsl`
float norm(float x, float a, float b) {
  return (x - a) / (b - a);
}`
)}

function _54(md){return(
md`### WebGL`
)}

function _axis(webgl,gl,m){return(
new class {
  constructor(size = 100) {
    this.model = webgl.compileModel(gl, {
      attribs: {
        position: {
          data: new Float32Array([
            0, 0, 0, 1, 0, 0, // x
            0, 0, 0, 0, 1, 0, // y
            0, 0, 0, 0, 0, 1  // z
          ])
        },
        color: {
          data: new Float32Array([
            1, 0, 0, 1, 0, 0, // r
            0, 1, 0, 0, 1, 0, // g
            0, 0, 1, 0, 0, 1, // b
          ]),
          size: 3
        }
      },
      mode: webgl.DrawMode.LINES, num: 6, uniforms: {},
      shader: webgl.defShader(gl, {
        vs: `
void main() {
  v_color = color;
  gl_Position = proj * view * model * vec4(position, 1.0);
}`,
        fs: `
void main() {
  fragColor = vec4(v_color, 1.0);
}`,
        declPrefixes: { v: "v_" },
        attribs: {
          position: 'vec3',
          color: 'vec3'
        },
        varying: {
          color: "vec3",
        },
        uniforms: {
          model: ["mat4", m.scale44([], size)],
          proj: "mat4",
          view: "mat4",
        },
        state: {
          depth: true
        }
      })
    });
  }
  
  draw(camera) {
    this.model.uniforms.view = camera.view;
    this.model.uniforms.proj = camera.proj;
    webgl.draw(this.model);
  }
}
)}

function _camera(View,Camera,gl,v,raf,tween){return(
new class extends View {
  constructor() {
    super();
    this.cam = new Camera({
      viewport: [gl.drawingBufferWidth, gl.drawingBufferHeight],
      eye: [0, 0, 0.01],
      near: 0.01, far: 200
    });
    Object.defineProperties(this, {
      _locked: { value: false, writable: true }
    });
  }
  
  orbitStart(x, y) {
    if (this._locked) return;
    this.cam.orbitStart(x, y);
  }
  
  orbitUpdate(x, y) {
    if (this._locked) return;
    this.cam.orbitUpdate(x, y);
    this.notify();
  }
  
  panStart(x, y) {
    if (this._locked) return;
    this.cam.panStart(x, y);
  }
  
  panUpdate(x, y) {
    if (this._locked) return;
    this.cam.panUpdate(x, y);
    this.notify();
  }
  
  zoom(delta, min, max) {
    if (this._locked) return;
    this.cam.zoom(delta, min, max);
    this.notify();
  }
  
  panZoom(x, y, delta, min, max) {
    if (this._locked) return;
    this.cam.panZoom(x, y, delta, min, max);
    this.notify();
  }
  
  goto({ eye = null, target = null, up = null }, speed = 0.08, eps = 1e-5) {
    this.cancel && this.cancel();
    const eye0 = v.copy(this.cam.eye);
    const target0 = v.copy(this.cam.target);
    const up0 = v.copy(this.cam.up);
    this._locked = true;

    this.cancel = raf(tween(t => {
      if (eye) this.cam.eye = v.mixN(this.cam.eye, eye0, eye, t);
      if (target) this.cam.target = v.mixN(this.cam.target, target0, target, t);
      if (up) this.cam.up = v.mixN(this.cam.up, up0, up, t);
      this.notify();
    },  speed, eps), () => {
      this._locked = false;
    });
  }
  
  get value() {
    return this.cam;
  }
}
)}

function _Camera(v,m,math){return(
class Camera {
  constructor(opts) {
    const {
      viewport, eye, target = v.ZERO3, up = v.Y3,
      fov = 45, near = 0.01, far = 50,
    } = opts;
    
    this.proj = m.perspective([], fov, viewport[0] / viewport[1], near, far);
    
    this.tanFov = Math.tan(fov * math.DEG2RAD / 2);
    
    Object.defineProperties(this, {
      _viewport: { value: viewport, writable: true },
      _radius: { value: Math.min(...viewport), writable: true },
      _center: { value: v.mulN([], viewport, 0.5), writable: true },
      
      _eye: { value: eye, writable: true },
      _up0: { value: v.copy(up), writable: true },
      _target: { value: v.copy(target), writable: true },

      _up: { value: [] }, _viewDir: { value: [] }, _down: { value: [] },
      
      _delta: { value: [] }, _side: { value: [] }, _axis: { value: [] },
      _vd: { value: [] }, _rot: { value: [] },
      
      _view: { value: m.lookAt([], eye, target, up) },
      
      _grabEye: { value: [] },
      _grabTarget: { value: [] },
      
      _dirty: { value: true, writable: true }
    });
  }
  
  _spherePos(out, x, y) {
    v.divN3(null, v.setC3(out, x - this._center[0], y - this._center[1], 0), this._radius);
    const mag = v.magSq3(out);
    return mag > 1.0 ? v.normalize(null, out) : (out[2] = Math.sqrt(1.0 - mag), out);
  }
  
  get eye() { return this._eye; }
  set eye(val) { this._eye = val; this._dirty = true; return this; }
  get target() { return this._target; }
  set target(val) { this._target = val; this._dirty = true; return this; }
  get up() { return this._up0; }
  set up(val) { this._up0 = val; this._dirty = true; return this; }
  
  get view() {
    if (!this._dirty) return this._view;
    
    this._dirty = false;
    return m.lookAt(this._view, this.eye, this.target, this.up);
  }
  
  orbitStart(x, y) {
    this._spherePos(this._down, x, y);
    
    v.set3(this._up, this.up);
    v.sub3(this._viewDir, this.eye, this.target);
    
    return this;
  }
  
  orbitUpdate(x, y, speed = 5) {
    const [dx, dy] = v.sub3(null, this._spherePos(this._delta, x, y), this._down);
    
    v.cross3(this._side, this._up, this._viewDir);
    v.normalize(null, this._side, dx);
    v.mulN3(this._axis, this._up, -dy);
    v.add3(null, this._axis, this._side);
    v.normalize(null, v.cross3(null, this._axis, this._viewDir));

    m.rotationAroundAxis33(this._rot, this._axis, v.magSq3(this._delta) * speed);
    m.mulV33(this.up, this._rot, this._up);
    v.add3(this.eye, this.target, m.mulV33(this._vd, this._rot, this._viewDir));
    
    this._dirty = true;
    return this;
  }
  
  panStart(x, y) {
    v.setC2(this._down, x, y);
    v.set3(this._grabEye, this.eye);
    v.set3(this._grabTarget, this.target);
    
    return this;
  }
  
  panUpdate(x, y) {
    const [dx, dy] = v.submN2(null, [x, y], this._down, 2 / this._radius);
    
    const viewDir = v.sub3(this._vd, this.target, this.eye);
    const dist = v.mag(viewDir) * this.tanFov;

    v.normalize(null, viewDir);
    const side = v.normalize(null, v.cross3(this._side, this.up, viewDir), dist * dx);
    const up = v.normalize(this._up, this.up, dist * dy);
    
    v.add3(null, v.add3(this.eye, this._grabEye, up), side);
    v.add3(null, v.add3(this.target, this._grabTarget, up), side);

    this._dirty = true;
    return this;
  }
  
  zoom(delta, min = 0.1, max = 10.0) {
    v.sub3(this._vd, this.target, this.eye);
    const mag = math.clamp(Math.pow(Math.E, delta) * v.mag(this._vd), min, max);
    v.sub3(this.eye, this.target, v.normalize(null, this._vd, mag));
    
    this._dirty = true;
    return this;
  }
  
  panZoom(x, y, delta, min, max) {
    const d = v.submN2(null, [x, y], this._center, -2 * this.tanFov / this._radius);
    const a0 = v.mag(v.sub3(this._vd, this.target, this.eye));
    this.zoom(delta, min, max);
    const viewDir = v.sub3(this._vd, this.target, this.eye);
    const [dx, dy] = v.mulN2(null, d, a0 - v.mag(viewDir));

    v.normalize(null, viewDir);
    const side = v.normalize(null, v.cross3(this._side, this.up, viewDir), dx);
    const up = v.normalize(this._up, this.up, dy);

    v.add3(null, v.add3(null, this.target, up), side);
    v.add3(null, v.add3(null, this.eye, up), side);
    
    this._dirty = true;
    return this;
  }
}
)}

function _58(md){return(
md`### Misc`
)}

function _ID(){return(
{
  encode: val => {
    const r = ( val >> 16 & 0xff ) / 255;
    const g = ( val >> 8 & 0xff ) / 255;
    const b = ( val & 0xff ) / 255;
    return [r, g, b];
  },
  decode: ([r, g, b]) => (r << 16) | (g << 8) | b
}
)}

function _hub(Generators,rs){return(
Generators.disposable(new class {
  constructor() {
    this.stream = rs.pubsub({ topic: e => e.id });
  }
  
  publish(id, event) {
    this.stream.next({ id, event });
  }

  subscribe(id, sub) {
    const s = this.stream.subscribeTopic(id, {
      next({ event }) { sub.next(event); },
      done: sub.done,
      error: sub.error
    });
    return () => this.stream.unsubscribeTopic(id, s.wrapped);
  }
  
  done() {
    this.stream.done();
  }
}, x => x.done())
)}

function _reportProgress(hub){return(
name => (current, total) => hub.publish(name, { current, total })
)}

function _loadingProgress(d3,html,hub,invalidation){return(
() => {
  const elm = d3.select(html`<div style="font: var(--monospace-font);">
    <div style="text-align:center;">Loading</div>
    <div><span>glyph set 1: </span><span id="part01">waiting...</span></div>
    <div><span>glyph set 2: </span><span id="part02">waiting...</span></div>
    <div><span>glyph set 3: </span><span id="part03">waiting...</span></div>
    <div><span>glyph set 4: </span><span id="part04">waiting...</span></div>
</div>`);
  const kb = x => x / 1024 | 0;
  const setup = name => {
    const el = elm.select(`#${name}`);
    const unsub0 = hub.subscribe(`${name}.download`, {
      next({ current, total }) { el.text(`downloaded ${kb(current)} / ${kb(total)} kb`); }
    });
    const unsub1 = hub.subscribe(`${name}.build`, {
      next({ current, total }) { el.text(`built ${current} / ${total} glyphs`); }
    });
    return [unsub0, unsub1];
  };
  const unsub0 = setup("part01");
  const unsub1 = setup("part02");
  const unsub2 = setup("part03");
  const unsub3 = setup("part04");
  
  const unsub4 = hub.subscribe("slugs", {
    next() { elm.style("display", "none"); }
  });
  
  invalidation.then(() => {
    unsub0.forEach(x => x());
    unsub1.forEach(x => x());
    unsub2.forEach(x => x());
    unsub3.forEach(x => x());
    unsub4();
  });
  return elm.node();
}
)}

function _63(md){return(
md`### Utils`
)}

function _dracoDecode(draco)
{
  function decodeBuffer(decoder, data) {
    const buffer = new draco.DecoderBuffer();
    buffer.Init(new Int8Array(data), data.byteLength);
    const geometryType = decoder.GetEncodedGeometryType(buffer);
    
    let geometry, status;
		if (geometryType === draco.TRIANGULAR_MESH) {
			geometry = new draco.Mesh();
			status = decoder.DecodeBufferToMesh(buffer, geometry);
		} else if (geometryType === draco.POINT_CLOUD) {
			geometry = new draco.PointCloud();
			status = decoder.DecodeBufferToPointCloud(buffer, geometry);
		} else {
      draco.destroy(decoder);
      draco.destroy(geometry);
      draco.destroy(buffer);
			throw new Error('Unknown geometry type.');
		}

		if (!status.ok() || geometry.ptr === 0) {
      draco.destroy(decoder);
      draco.destroy(geometry);
      draco.destroy(buffer);
			throw new Error('Decoding failed: ' + status.error_msg());
		}
    
    geometry.type = geometryType;
    
    draco.destroy(buffer);
    return geometry;
  }

  const ATTRIB_ID = {
    position: draco.POSITION,
    normal: draco.NORMAL,
    color: draco.COLOR,
    uv: draco.TEX_COORD
  };
  
  const ARRAY = {
    [draco.DT_FLOAT32]: Float32Array,
    [draco.DT_FLOAT64]: Float64Array,
    [draco.DT_INT8]: Int8Array,
    [draco.DT_INT16]: Int16Array,
    [draco.DT_INT32]: Int32Array,
    [draco.DT_UINT8]: Uint8Array,
    [draco.DT_UINT16]: Uint16Array,
    [draco.DT_UINT32]: Uint32Array
  };
  
  const HEAP = {
    [draco.DT_FLOAT32]: draco.HEAPF32,
    [draco.DT_FLOAT64]: draco.HEAPF64,
    [draco.DT_INT8]: draco.HEAP8,
    [draco.DT_INT16]: draco.HEAP16,
    [draco.DT_INT32]: draco.HEAP32,
    [draco.DT_UINT8]: draco.HEAPU8,
    [draco.DT_UINT16]: draco.HEAPU16,
    [draco.DT_UINT32]: draco.HEAPU32
  };

  function decodeAttribute(decoder, geometry, key, type) {
    let attId;
    if (key in ATTRIB_ID) {
      attId = decoder.GetAttributeId(geometry, ATTRIB_ID[key]);
    } else {
      attId = decoder.GetAttributeIdByName(geometry, key);
    }
    if (attId == -1)
      return null;

    const attribute = decoder.GetAttribute(geometry, attId);
		const numValues = geometry.num_points() * attribute.num_components();
		const byteLength = numValues * ARRAY[type].BYTES_PER_ELEMENT;
    
		const ptr = draco._malloc(byteLength);
		decoder.GetAttributeDataArrayForAllPoints(geometry, attribute, type, byteLength, ptr);
		const array = new ARRAY[type](HEAP[type].buffer, ptr, numValues).slice();
		draco._free(ptr);

		return array;
	}
  
  function decodeIndices(decoder, geometry) {
    const numIndices = geometry.num_faces() * 3;
    const byteLength = numIndices * 4;

    const ptr = draco._malloc(byteLength);
    decoder.GetTrianglesUInt32Array(geometry, byteLength, ptr);
    const index = new Uint32Array(draco.HEAPF32.buffer, ptr, numIndices).slice();
    draco._free(ptr);
    
    return index;
  }
  
  return (data, attribs) => {
    const decoder = new draco.Decoder();
    const geometry = decodeBuffer(decoder, data);
    
    const mesh = {};
    for (const key in attribs) {
      mesh[key] = decodeAttribute(decoder, geometry, key, attribs[key]);
    }
  
    if (geometry.type == draco.TRIANGULAR_MESH) {
      mesh["indices"] = decodeIndices(decoder, geometry);
    }
    
    draco.destroy(geometry);
    draco.destroy(decoder);
    
    return mesh;
  }
}


function _decompress(zstd)
{
  const ZSTD_isError = zstd.cwrap('ZSTD_isError', 'number', ['number']);
  const ZSTD_getFrameContentSize = zstd.cwrap('ZSTD_getFrameContentSize','number', ['array', 'number']);
  const ZSTD_decompress = zstd.cwrap('ZSTD_decompress', 'number', ['number', 'number', 'array', 'number']);

  return data => {
    const contentSize = ZSTD_getFrameContentSize(data, data.length);
    if (!contentSize) {
      throw new Error('zstd: Unable to get frame content size.');
    }

    const heap = zstd._malloc(contentSize);
    try {
      const decompressRc = ZSTD_decompress(heap, contentSize, data, data.length);
      if (ZSTD_isError(decompressRc) || decompressRc != contentSize)
        throw new Error('zstd: Unable to decompress.');

      return new Uint8Array(zstd.HEAPU8.buffer, heap, contentSize);
    } finally {
      zstd._free(heap);
    }
  };
}


function _fetchProgress()
{
  function concat(arrays) {
    let i = 0, n = 0;
    for (const a of arrays) n += a.length;
    const concat = new Uint8Array(n);
    for (let a of arrays) concat.set(a, i), i += a.length;
    return concat.buffer;
  }
    
  return (url, init, progress) => {
    let cur = 0;
    return fetch(url, init).then(async response => {
      const cl = response.headers.get("content-length");
      const tot = cl ? parseInt(cl) : -1;
      const reader = response.body.getReader();
      const values = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        cur += value.length;
        progress && progress(cur, tot);
        values.push(value);
      }
      return concat(values);
    });
  };
}


function _debounce(Generators){return(
function debounce(input, delay = 100) {
  return Generators.observe(notify => {
    let timer = null;
    let value;
    
    function delayed() {
      timer = null;
      if (value === input.value) return;
      notify(value = input.value);
    }

    function inputted() {
      if (timer !== null) return;
      notify(value = input.value);
      timer = setTimeout(delayed, delay);
    }

    input.addEventListener("input", inputted), inputted();
    return () => input.removeEventListener("input", inputted);
  });
}
)}

function _tween(math){return(
(f, speed = 0.08, eps = 1e-6) => {
  let t = 0;
  return () => {
    f(t);
    t += (1 - t) * speed;
    return !math.eqDelta(t, 1, eps);
  };
}
)}

function _raf(){return(
(f, done) => {
  let handle = requestAnimationFrame(function frame(t) {
    if (f(t) === false) {
      done && done();
    } else {
      handle = requestAnimationFrame(frame);
    }
  });
  return () => cancelAnimationFrame(handle);
}
)}

function _approach(math){return(
function* approach(start, end, speed = 0.08, eps = 1e-4) {
  start = start !== undefined ? start : end;
  yield start;
  while (!math.eqDelta(start, end, eps)) {
    yield start += (end - start) * speed;
  }
}
)}

function _View(Generators){return(
class View {
  static create(initialize) {
    return new Promise(resolve => {
      const view = new View();
      let resolved = false;
      const dispose = initialize((val, state = val) => {
        view.value = val;
        view.state = state;
        if (!resolved) {
          resolved = true;
          resolve(Generators.disposable(view, () => {
            dispose && dispose();
            view.dispose();
          }));
        };
      })
    });
  }
  
  static state(view, defaults) {
    return !!view ? view.state : defaults;
  }
  
  static subs(unsubs) {
    return () => unsubs.forEach(unsub => unsub());
  }
  
  constructor(value) {
    Object.defineProperties(this, {
      _value: { value, writable: true },
      _list: { value: {}, writable: true },
      _subs: { value: [], writable: true },
    });
    
    const dispatch = e => this._subs.forEach(sub => sub(e.value));
    this.addEventListener('input', dispatch);
    this.dispose = () => this.removeEventListener("input", dispatch);
  };
  
  get value() { return this._value; }
  set value(val) { this._value = val; this.notify(); }
  
  observe(sub) {
    this._subs.push(sub);
    sub(this.value); // give out the current value on subscription
    return () => this._subs = this._subs.filter(l => l !== sub);
  }
  notify() {
    this.dispatchEvent({ type: "input", value: this.value });
  }
  addEventListener(type, listener) {
    if (!this._list[type]) this._list[type] = [];
    if (this._list[type].includes(listener)) return;
    this._list[type] = [listener].concat(this._list[type]);
  }
  removeEventListener(type, listener) {
    this._list[type] = this._list[type].filter(l => l !== listener);
  }
  dispatchEvent(event) {
    if (!this._list[event.type]) return;
    const p = Promise.resolve(event);
    this._list[event.type].forEach(l => p.then(l));
  }
}
)}

function _fps()
{
  let t = performance.now(), delta = 0;
  const last = [t];
  return (cell, size = 10) => {
    last.length >= size && last.shift();
    t = performance.now();
    delta = (t - last[0]) / last.length;
    last.push(t);
    return Math.round(1000 / delta);
  }
}


function _73(md){return(
md`### Appendix`
)}

function _webgl(require){return(
require("https://bundle.run/@thi.ng/webgl@4.0.13")
)}

function _m(require){return(
require("https://bundle.run/@thi.ng/matrices@0.6.56")
)}

function _v(require){return(
require("https://bundle.run/@thi.ng/vectors@6.0.0")
)}

function _math(require){return(
require("https://bundle.run/@thi.ng/math@4.0.0")
)}

function _rs(require){return(
require("https://bundle.run/@thi.ng/rstream@6.0.8")
)}

async function _draco(require)
{
  const PREFIX = "https://www.gstatic.com/draco/versioned/decoders/1.4.1/";
  const wasmBinary = await fetch(`${PREFIX}/draco_decoder.wasm`).then(resp => {
    if (!resp.ok)
      throw new Error("HTTP error, status = " + resp.status);
    return resp.arrayBuffer();
  });

  const wrapper = await require(`${PREFIX}/draco_wasm_wrapper.js`);
  return new Promise(resolve => wrapper({ wasmBinary }).then(draco => {
    delete draco.then; // tell Observable not wait for this;
    resolve(draco);
  }));
}


function _zstd(require){return(
require("https://bundle.run/zstd-wasm@0.0.6")
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  function toString() { return this.url; }
  const fileAttachments = new Map([
    ["embedding.drc", {url: new URL("./files/966deceabc5f4c1e97bbf36a18b53c8f486ab78cbf9e69f234fc93b40688a5a73b3b4d33e7f10eccf217812bee88c672b5177b48a1e7a03d075c900cabc41250.bin", import.meta.url), mimeType: "application/octet-stream", toString}],
    ["SourceHanSerifTC-Bold_glyphs.json@1.zst", {url: new URL("./files/0084dda83482fb07dfc09ed8bf640b90650984a97b6f05a3f9f0bc8352994fdc2426ff036a301a8bfa6cf634e4e02275051dfc730d6970ffd1906ec5ab5aac77.bin", import.meta.url), mimeType: "application/octet-stream", toString}],
    ["SourceHanSerifTC-Bold_data_3.bin", {url: new URL("./files/8f12e06f1485a104e8a6dd672c3bef95ad52f7159adf3bcbae4431c41fb4652e18295f28408e6dc850b15a7802883a8411df020fd0c00b9595c88a91170607c0.bin", import.meta.url), mimeType: "application/octet-stream", toString}],
    ["SourceHanSerifTC-Bold_data_1.bin", {url: new URL("./files/72496fa18675b8f41047f8fe5fc51dec25562cc47a848d88437c61ec2fa98c651918c41d242d81f3037c16bf1335dea1fa9e1fe66564d6ccfd33de74ca33986f.bin", import.meta.url), mimeType: "application/octet-stream", toString}],
    ["SourceHanSerifTC-Bold_data_2.bin", {url: new URL("./files/e217edd7c33f5ff81f94aa00580e1df84448936141d6852f72119e906273f41611da592ef5a21f092dce3bcbff23460c744f5c9ba4b6f94b71b8364f3af41f95.bin", import.meta.url), mimeType: "application/octet-stream", toString}],
    ["SourceHanSerifTC-Bold_data_0.bin", {url: new URL("./files/6ff04b96387b13b6fc502fb3f58a30e2198d6894b1ce7cbb2f41689e9abcb46bac24d68fd6447dac38457dd3a4c4ff4f6de3f90bc920f62662e3f32680f3fc42.bin", import.meta.url), mimeType: "application/octet-stream", toString}],
    ["fig_ae_loss.png", {url: new URL("./files/83c6616fd2406f88a1f4cd03749928f4b4d2becb0a668e0c7cd72b497b0dd37621bbeb6eb955f2880390a1056c69c4774cfedcd62a8b1f138c75bd2a24b1255f.png", import.meta.url), mimeType: "image/png", toString}],
    ["fig_ae_res.png", {url: new URL("./files/34cfb33ea2b4860134dd8e914b7e5598a01977ca0608277347c742f16755f3cf27c943b1744b2c5f209390bcd44fd66758c181a29ecdc9473bbfbb92084132b7.png", import.meta.url), mimeType: "image/png", toString}],
    ["fig_geomloss_grid.png", {url: new URL("./files/013978d4b79e902d79064d04ffac674e365369508de43d3a885b874b1b275da8e524544387a7b6bd18cb304d4264610658493acd017d849967db2d98d0408b85.png", import.meta.url), mimeType: "image/png", toString}],
    ["fig_mde_3d.png", {url: new URL("./files/692527117e7a6b32cceec77e4d7de44a7b9e09ea7f6b06cead7883daad515cbb2ebb2fe5340bc98c8ac791ff71aa9c865118d427f23e9da305ba356691b3d4e2.png", import.meta.url), mimeType: "image/png", toString}],
    ["fig_mde_2d.png", {url: new URL("./files/7e63fad3fe3e2ad8e1d9ff527ae6245d3a408223a2a21fb98a8b66dc7d1622426ed3d38a3704eebd3897437b78911398fefb0c876f55f292b00520229eebab8b.png", import.meta.url), mimeType: "image/png", toString}],
    ["fig_geomloss_lap.png", {url: new URL("./files/fef88fef4e74efc3ccdf34b6887634880c03900d0b4865b6a63271367e9117844bb76ca50f01ed7edcd23a317467ce639b52e4895cb2a87d5c8ca21e109236ae.png", import.meta.url), mimeType: "image/png", toString}],
    ["fig_slug.png", {url: new URL("./files/0bb11f199de7632a230f7af41a0e57a5acaaef8fc13037baa1ec596966dc5c99134e27081d7babd422653f0f2a05ce4df2c7db5efe8db1a6b8b0ed2a96f52ae1.png", import.meta.url), mimeType: "image/png", toString}]
  ]);
  main.builtin("FileAttachment", runtime.fileAttachments(name => fileAttachments.get(name)));
  main.variable(observer()).define(["md"], _1);
  main.variable(observer()).define(["md"], _2);
  main.variable(observer("viewof glyphIdx")).define("viewof glyphIdx", ["d3","glyphSets","html"], _glyphIdx);
  main.variable(observer("glyphIdx")).define("glyphIdx", ["Generators", "viewof glyphIdx"], (G, _) => G.input(_));
  main.variable(observer("viewof is3d")).define("viewof is3d", ["Inputs"], _is3d);
  main.variable(observer("is3d")).define("is3d", ["Generators", "viewof is3d"], (G, _) => G.input(_));
  main.variable(observer("viewof gl")).define("viewof gl", ["DOM","width","html","loadingProgress"], _gl);
  main.variable(observer("gl")).define("gl", ["Generators", "viewof gl"], (G, _) => G.input(_));
  main.variable(observer()).define(["md"], _6);
  main.variable(observer()).define(["md","html","FileAttachment"], _7);
  main.variable(observer()).define(["md","html","FileAttachment"], _8);
  main.variable(observer()).define(["md","html","FileAttachment"], _9);
  main.variable(observer()).define(["md","html","FileAttachment"], _10);
  main.variable(observer()).define(["md"], _11);
  main.variable(observer("render")).define("render", ["gl","picking","camera","t","atlases","hover","selected"], _render);
  main.variable(observer("t")).define("t", ["atlases","approach","is3d"], _t);
  main.variable(observer("hover")).define("hover", ["Generators","d3","gl","picking"], _hover);
  main.variable(observer("selected")).define("selected", ["debounce","viewof glyphIdx"], _selected);
  main.variable(observer()).define(["atlases","d3","gl","picking","viewof glyphIdx","invalidation"], _16);
  main.variable(observer("focus")).define("focus", ["atlases","viewof free","viewof camera","v","is3d","embedding","selected"], _focus);
  main.variable(observer()).define(["viewof camera","d3","gl","is3d","viewof free","invalidation"], _18);
  main.variable(observer("freeView")).define("freeView", ["viewof free","glyphIdx"], _freeView);
  main.variable(observer("viewof free")).define("viewof free", ["View"], _free);
  main.variable(observer("free")).define("free", ["Generators", "viewof free"], (G, _) => G.input(_));
  main.variable(observer()).define(["md"], _21);
  main.variable(observer("embedding")).define("embedding", ["FileAttachment","dracoDecode","draco","makeGrid","d3"], _embedding);
  main.variable(observer("embeddings")).define("embeddings", ["subEmbedding","embedding"], _embeddings);
  main.variable(observer("subEmbedding")).define("subEmbedding", _subEmbedding);
  main.variable(observer("makeGrid")).define("makeGrid", ["d3"], _makeGrid);
  main.variable(observer()).define(["md"], _26);
  main.variable(observer("slugReady")).define("slugReady", ["atlases","hub"], _slugReady);
  main.variable(observer("font0")).define("font0", ["buildSlug","FileAttachment","glyphSets","Font"], _font0);
  main.variable(observer("font1")).define("font1", ["buildSlug","FileAttachment","glyphSets","Font"], _font1);
  main.variable(observer("font2")).define("font2", ["buildSlug","FileAttachment","glyphSets","Font"], _font2);
  main.variable(observer("font3")).define("font3", ["buildSlug","FileAttachment","glyphSets","Font"], _font3);
  main.variable(observer("picking")).define("picking", ["atlases","Picking","embedding"], _picking);
  main.variable(observer("atlases")).define("atlases", ["Atlas","embeddings","font0","font1","font2","font3"], _atlases);
  main.variable(observer("glyphSets")).define("glyphSets", ["FileAttachment","decompress"], _glyphSets);
  main.variable(observer("buildSlug")).define("buildSlug", ["fetchProgress","reportProgress","makeSlug","decodeCurvesData"], _buildSlug);
  main.variable(observer("makeSlug")).define("makeSlug", ["rs"], _makeSlug);
  main.variable(observer("Picking")).define("Picking", ["webgl","gl","d3","ID","m"], _Picking);
  main.variable(observer("Atlas")).define("Atlas", ["webgl","gl","d3","ID","moddiv","tangentAt","gradAt","locAt","getRootCode","traceRay","getCurveCoord","traceRayBandH","traceRayBandV","aastep","frame","norm","m"], _Atlas);
  main.variable(observer("Font")).define("Font", ["webgl","gl"], _Font);
  main.variable(observer("decodeCurvesData")).define("decodeCurvesData", ["decompress"], _decodeCurvesData);
  main.variable(observer("viewof getRootCode")).define("viewof getRootCode", ["glsl"], _getRootCode);
  main.variable(observer("getRootCode")).define("getRootCode", ["Generators", "viewof getRootCode"], (G, _) => G.input(_));
  main.variable(observer("viewof traceRay")).define("viewof traceRay", ["glsl"], _traceRay);
  main.variable(observer("traceRay")).define("traceRay", ["Generators", "viewof traceRay"], (G, _) => G.input(_));
  main.variable(observer("viewof getCurveCoord")).define("viewof getCurveCoord", ["glsl"], _getCurveCoord);
  main.variable(observer("getCurveCoord")).define("getCurveCoord", ["Generators", "viewof getCurveCoord"], (G, _) => G.input(_));
  main.variable(observer("viewof traceRayBandH")).define("viewof traceRayBandH", ["glsl"], _traceRayBandH);
  main.variable(observer("traceRayBandH")).define("traceRayBandH", ["Generators", "viewof traceRayBandH"], (G, _) => G.input(_));
  main.variable(observer("viewof traceRayBandV")).define("viewof traceRayBandV", ["glsl"], _traceRayBandV);
  main.variable(observer("traceRayBandV")).define("traceRayBandV", ["Generators", "viewof traceRayBandV"], (G, _) => G.input(_));
  main.variable(observer("viewof moddiv")).define("viewof moddiv", ["glsl"], _moddiv);
  main.variable(observer("moddiv")).define("moddiv", ["Generators", "viewof moddiv"], (G, _) => G.input(_));
  main.variable(observer("viewof pointAt")).define("viewof pointAt", ["glsl"], _pointAt);
  main.variable(observer("pointAt")).define("pointAt", ["Generators", "viewof pointAt"], (G, _) => G.input(_));
  main.variable(observer("viewof tangentAt")).define("viewof tangentAt", ["glsl"], _tangentAt);
  main.variable(observer("tangentAt")).define("tangentAt", ["Generators", "viewof tangentAt"], (G, _) => G.input(_));
  main.variable(observer("viewof gradAt")).define("viewof gradAt", ["glsl"], _gradAt);
  main.variable(observer("gradAt")).define("gradAt", ["Generators", "viewof gradAt"], (G, _) => G.input(_));
  main.variable(observer("viewof locAt")).define("viewof locAt", ["glsl"], _locAt);
  main.variable(observer("locAt")).define("locAt", ["Generators", "viewof locAt"], (G, _) => G.input(_));
  main.variable(observer("viewof aastep")).define("viewof aastep", ["glsl"], _aastep);
  main.variable(observer("aastep")).define("aastep", ["Generators", "viewof aastep"], (G, _) => G.input(_));
  main.variable(observer("viewof frame")).define("viewof frame", ["glsl"], _frame);
  main.variable(observer("frame")).define("frame", ["Generators", "viewof frame"], (G, _) => G.input(_));
  main.variable(observer("viewof norm")).define("viewof norm", ["glsl"], _norm);
  main.variable(observer("norm")).define("norm", ["Generators", "viewof norm"], (G, _) => G.input(_));
  main.variable(observer()).define(["md"], _54);
  main.variable(observer("axis")).define("axis", ["webgl","gl","m"], _axis);
  main.variable(observer("viewof camera")).define("viewof camera", ["View","Camera","gl","v","raf","tween"], _camera);
  main.variable(observer("camera")).define("camera", ["Generators", "viewof camera"], (G, _) => G.input(_));
  main.variable(observer("Camera")).define("Camera", ["v","m","math"], _Camera);
  main.variable(observer()).define(["md"], _58);
  main.variable(observer("ID")).define("ID", _ID);
  main.variable(observer("hub")).define("hub", ["Generators","rs"], _hub);
  main.variable(observer("reportProgress")).define("reportProgress", ["hub"], _reportProgress);
  main.variable(observer("loadingProgress")).define("loadingProgress", ["d3","html","hub","invalidation"], _loadingProgress);
  main.variable(observer()).define(["md"], _63);
  main.variable(observer("dracoDecode")).define("dracoDecode", ["draco"], _dracoDecode);
  main.variable(observer("decompress")).define("decompress", ["zstd"], _decompress);
  main.variable(observer("fetchProgress")).define("fetchProgress", _fetchProgress);
  main.variable(observer("debounce")).define("debounce", ["Generators"], _debounce);
  main.variable(observer("tween")).define("tween", ["math"], _tween);
  main.variable(observer("raf")).define("raf", _raf);
  main.variable(observer("approach")).define("approach", ["math"], _approach);
  main.variable(observer("View")).define("View", ["Generators"], _View);
  main.variable(observer("fps")).define("fps", _fps);
  main.variable(observer()).define(["md"], _73);
  const child1 = runtime.module(define1);
  main.import("glsl", child1);
  main.variable(observer("webgl")).define("webgl", ["require"], _webgl);
  main.variable(observer("m")).define("m", ["require"], _m);
  main.variable(observer("v")).define("v", ["require"], _v);
  main.variable(observer("math")).define("math", ["require"], _math);
  main.variable(observer("rs")).define("rs", ["require"], _rs);
  main.variable(observer("draco")).define("draco", ["require"], _draco);
  main.variable(observer("zstd")).define("zstd", ["require"], _zstd);
  return main;
}
