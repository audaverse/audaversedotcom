function _1(md){return(
md`# GLSL Chunk Tag`
)}

function _2(md){return(
md`A simple widget to modularize shaders in notebooks.

Inspired by [Jed Fox](https://observablehq.com/@j-f1/)'s [CSS template tag](https://observablehq.com/@j-f1/css-template-tag), and to achieve the usage experiences of [glslify](https://github.com/glslify/glslify) or three.js [shader chunks](https://github.com/mrdoob/three.js/tree/dev/src/renderers/shaders).`
)}

function _3(md){return(
md`---
## Usage`
)}

function _4(md){return(
md`
~~~js
import {glsl} from "@stwind/glsl-chunk-tag"
~~~`
)}

function _5(md){return(
md`### GLSL Chunks`
)}

function _fade(glsl){return(
glsl`
float fade(float t)
{
  return t*t*t*(t*(t*6.-15.)+10.);
}`
)}

function _perm(glsl){return(
glsl`
float perm(float x)
{
  return texture2D(permutation,vec2(x/256.,0)).r*256.;
}`
)}

function _grad(glsl){return(
glsl`
float grad(float x, vec3 p)
{
  return dot(texture2D(grads,vec2(x,0)).rgb,p);
}`
)}

function _pnoise(glsl){return(
glsl.open`
float pnoise(vec3 p)
{
  vec3 P = mod(floor(p),256.);
  p -= floor(p);
  vec3 f = vec3(fade(p.x), fade(p.y), fade(p.z));
  float A  = perm(P.x) + P.y;
  float AA = perm(A) + P.z;
  float AB = perm(A + 1.) + P.z;
  float B  = perm(P.x + 1.) + P.y;
  float BA = perm(B) + P.z;
  float BB = perm(B + 1.) + P.z;

  return mix(
    mix(
      mix(grad(perm(AA), p),
          grad(perm(BA), p + vec3(-1., 0.,0.)), f.x),
      mix(grad(perm(AB), p + vec3( 0.,-1.,0.)),
          grad(perm(BB), p + vec3(-1.,-1.,0.)), f.x), 
      f.y),
    mix(
      mix(grad(perm(AA + 1.), p + vec3( 0., 0.,-1.)),
          grad(perm(BA + 1.), p + vec3(-1., 0.,-1.)), f.x),
      mix(grad(perm(AB + 1.), p + vec3( 0.,-1.,-1.)),
          grad(perm(BB + 1.), p + vec3(-1.,-1.,-1.)), f.x), 
      f.y),
    f.z);
}`
)}

function _norm(glsl){return(
glsl`
float norm(float x, float a, float b) 
{
  return (x - a) / (b - a);
}`
)}

function _rotate2d(glsl){return(
glsl`
mat2 rotate2d(float angle)
{
  float s = sin(angle);
  float c = cos(angle);
  return mat2(c, -s, 
              s,  c);
}`
)}

function _wrap(glsl){return(
glsl`
vec2 wrap(vec2 p)
{
  return 1. - abs(1. - mod(p, 2.));
}`
)}

function _fbm(glsl){return(
glsl`
float fbm(vec3 p, int octaves, float decay, float shift) 
{
  float ret = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 5; i++) {
    if (i >= octaves) break;

    ret += pnoise(p + shift * float(i)) * amp;
    amp *= decay;
    p *= 2.0;
  }
  
  return ret;
}`
)}

function _disp(glsl){return(
glsl`
vec2 disp(vec2 p, float v, float r, float freq)
{
  float a = v * freq;
  return p + vec2(r * cos(a), (r + 0.3) * sin(a + PI * 0.25));
}`
)}

function _gradient(glsl){return(
glsl`
vec3 gradient(float t, vec3 a, vec3 b, vec3 c, vec3 d)
{
// http://www.iquilezles.org/www/articles/palettes/palettes.htm
  return a + b*cos( 6.28318*(c*t+d) );
}`
)}

function _16(md){return(
md`### Rendering`
)}

function _gl(DOM,width)
{
  const canvas = DOM.canvas(width, width * 0.524);
  canvas.value = canvas.getContext('webgl');
  return canvas;
}


function _render(gl,webgl,model,now,textures)
{
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);

  webgl.bind(model);
  
  const time = now % 1e6 * 4e-4;

  webgl.setUniforms(model, {
    scale: ['1f', 2.4],
    shift: ['1f', time],
    freq: ['1f', 13],
    octaves: ['1i', 3],
    iter: ['1i', 3],
    offset: ['1f', time * 2e-1],
    
    permutation: ['tex', [textures.perm, 0]],
    grads: ['tex', [textures.grads, 1]]
  });

  gl.drawArrays(gl.TRIANGLES, 0, 6);
  
  webgl.unbind();
}


function _model(webgl,fade,perm,grad,pnoise,norm,fbm,disp,wrap,gradient){return(
webgl.createObject({ 
  attribs: {
    position: { data: Float32Array.of(-1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1), size: 2 },
    uv: { data: Float32Array.of(0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1), size: 2 },
  },
  vs: `
precision highp int;
precision highp float;

attribute vec2 position;
attribute vec2 uv;

varying vec2 v_uv;

void main() {
  v_uv = uv;
  gl_Position = vec4(position, 0.0, 1.0);
}`,
  fs: `
precision highp int;
precision highp float;

const float SQRT05 = 0.707106;
const float PI = 3.14159265359;

uniform float scale;
uniform float shift;
uniform float amp;
uniform int octaves;
uniform float freq;
uniform int iter;
uniform float offset;
uniform sampler2D permutation;
uniform sampler2D grads;

varying vec2 v_uv;

${fade}
${perm}
${grad}
${pnoise}
${norm}
${fbm}
${disp}
${wrap}
${gradient}

void main() {
  vec2 uv = v_uv + offset;
  
  float step = freq / float(iter);
  for (int i = 0; i < 5; i++) {
    if (i >= iter) break;
    float v = fbm(vec3(uv * scale, 0.0), octaves, 0.5, shift);
    uv = disp(uv, v, 0.15, step);
  }

  uv = wrap(uv);
  float t = pow((uv.x + uv.y) / 2.0, 2.0) + 0.2;
  vec3 c = gradient(t, 
    vec3(0.0,0.5,0.5),vec3(0.0,0.5,0.5),vec3(0.0,0.538,0.328),vec3(0.0,0.5,0.678)
  );
  gl_FragColor = vec4(c, 1.);
}`
})
)}

function _20(md){return(
md`---
## Implementation`
)}

function _glsl(html,md)
{
  const interpolate = (strings, ...args) => {
    let s = '';
    for (let i = 0; i < strings.length; i++) {
      s += strings[i];
      if (i < args.length && args[i]) s += String(args[i]);
    }
    return s;
  };

  const ID = 'glsl-styles';

  const style = html`<style id="${ID}">
details.glsl summary { 
  list-style-type: none; /* hide the triangle in firefox */
  outline: none;
  cursor: pointer;
  user-select: none;
}

details.glsl summary::marker {
  display: none;
}

details.glsl summary:before {
  content: "";
  display: inline-block;
  width: 0;
  height: 0;
  border-top: 4px solid transparent;
  border-bottom: 4px solid transparent;
  border-left: 6px solid rgb(27,30,35);
  border-right: 0;
  margin-right: 1px;
}

details[open].glsl summary:before {
  border-left: 4px solid transparent;
  border-right: 4px solid transparent;
  border-top: 6px solid rgb(27,30,35);
  border-bottom: 0;
  margin-left: -2px;
}

details[open].glsl summary code.glsl span,
details[open].glsl summary code.glsl,
details[open].glsl summary {
  color: grey;
}
<style>`;

  const createFunction = opts => (...args) => {
    let string = interpolate(...args);
    if (!string.startsWith('\n')) string = '\n' + string;

    if (!document.getElementById(ID)) document.head.appendChild(style);

    const sig = string.match(/^\n*(.*)$/m)[1];
    const summary = md`~~~glsl\n${sig}~~~`;
    summary.style.display = 'inline-block';
    summary.style.margin = '0';
    summary.style.verticalAlign = 'middle';

    const el = html`<details ${opts.open ? 'open' : ''} class='glsl'>
      <summary> ${summary} &hellip;</summary>
      ${md`~~~glsl${string}~~~`}
    </details>`;
    el.value = string;
    return el;
  };

  const returnValue = createFunction({ open: false });
  returnValue.open = createFunction({ open: true });
  return returnValue;
}


function _22(md){return(
md`---
## WebGL Setup`
)}

function _textures(webgl,gl)
{
  const P = Uint8Array.of(151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180);
  const G = Uint8Array.of(1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,1, 1, 0, 0, -1, 1, -1, 1, 0, 0, -1, -1);
  
  return {
    perm: webgl.createTexture(P, P.length, 1, gl.LUMINANCE),
    grads: webgl.createTexture(G, G.length / 3, 1, gl.RGB),
  };
}


function _webgl(gl)
{
  const ext = gl.getExtension("OES_vertex_array_object");

  const loadShader = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`Error compiling shader: ${gl.getShaderInfoLog(shader)}`);
      gl.deleteShader(shader);
    }
    return shader;
  }
  
  const compileProgram = (vs, fs) => {
    const program = gl.createProgram();
    gl.attachShader(program, loadShader(gl.VERTEX_SHADER, vs));
    gl.attachShader(program, loadShader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`Couldn't link shader: ${gl.getProgramInfoLog(program)}`);
      gl.deleteProgram(program);
    }

    return program;
  }
  
  const createVAO = (program, attribs) => {
    const vao = ext.createVertexArrayOES();
    ext.bindVertexArrayOES(vao);

    Object.keys(attribs).forEach(name => {
      const { data, size, stride = 0, offset = 0 } = attribs[name];
      
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      
      const loc = gl.getAttribLocation(program, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
    });

    ext.bindVertexArrayOES(null);
    return vao;
  };
  
  const location = (program, name) => gl.getUniformLocation(program, name);
  const uniformSetters = {
    '1f': (program, name, value) => gl.uniform1f(location(program, name), value),
    '2f': (program, name, value) => gl.uniform2f(location(program, name), ...value),
    '1i': (program, name, value) => gl.uniform1i(location(program, name), value),
    'tex': (program, name, [tex, id]) => {
      gl.activeTexture(gl.TEXTURE0 + id);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(location(program, name), id);
    }
  };
  
  return {
    createObject: ({ attribs, vs, fs }) => {
      const program = compileProgram(vs, fs);
      const vao = createVAO(program, attribs);
      return { vao, program }
    },
    createTexture: (data, width, height, format = gl.RGBA, type = gl.UNSIGNED_BYTE) => {
      const tex = gl.createTexture();
      
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, format, width, height, 0, format, type, data);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      gl.bindTexture(gl.TEXTURE_2D, null);

      return tex
    },
    setUniforms: ({ program }, values) => {
      Object.entries(values).forEach(([k, v]) => uniformSetters[v[0]](program, k, v[1]));
    },
    bind: ({ vao, program }) => {
      gl.useProgram(program);
      ext.bindVertexArrayOES(vao);
    },
    unbind: () => {
      ext.bindVertexArrayOES(null);
    }
  }
}


function _25(md){return(
md`---
## References`
)}

function _26(md){return(
md`
* [Improved Noise reference implementation](https://mrl.nyu.edu/~perlin/noise/)
* [The Book of Shaders: Fractal Brownian Motion](https://thebookofshaders.com/13/)
  * [domain warping by Inigo Quilez](http://www.iquilezles.org/www/articles/warp/warp.htm)
* [Perlin Noise / Mike Bostock / Observable](https://observablehq.com/@mbostock/perlin-noise)
`
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer()).define(["md"], _1);
  main.variable(observer()).define(["md"], _2);
  main.variable(observer()).define(["md"], _3);
  main.variable(observer()).define(["md"], _4);
  main.variable(observer()).define(["md"], _5);
  main.variable(observer("viewof fade")).define("viewof fade", ["glsl"], _fade);
  main.variable(observer("fade")).define("fade", ["Generators", "viewof fade"], (G, _) => G.input(_));
  main.variable(observer("viewof perm")).define("viewof perm", ["glsl"], _perm);
  main.variable(observer("perm")).define("perm", ["Generators", "viewof perm"], (G, _) => G.input(_));
  main.variable(observer("viewof grad")).define("viewof grad", ["glsl"], _grad);
  main.variable(observer("grad")).define("grad", ["Generators", "viewof grad"], (G, _) => G.input(_));
  main.variable(observer("viewof pnoise")).define("viewof pnoise", ["glsl"], _pnoise);
  main.variable(observer("pnoise")).define("pnoise", ["Generators", "viewof pnoise"], (G, _) => G.input(_));
  main.variable(observer("viewof norm")).define("viewof norm", ["glsl"], _norm);
  main.variable(observer("norm")).define("norm", ["Generators", "viewof norm"], (G, _) => G.input(_));
  main.variable(observer("viewof rotate2d")).define("viewof rotate2d", ["glsl"], _rotate2d);
  main.variable(observer("rotate2d")).define("rotate2d", ["Generators", "viewof rotate2d"], (G, _) => G.input(_));
  main.variable(observer("viewof wrap")).define("viewof wrap", ["glsl"], _wrap);
  main.variable(observer("wrap")).define("wrap", ["Generators", "viewof wrap"], (G, _) => G.input(_));
  main.variable(observer("viewof fbm")).define("viewof fbm", ["glsl"], _fbm);
  main.variable(observer("fbm")).define("fbm", ["Generators", "viewof fbm"], (G, _) => G.input(_));
  main.variable(observer("viewof disp")).define("viewof disp", ["glsl"], _disp);
  main.variable(observer("disp")).define("disp", ["Generators", "viewof disp"], (G, _) => G.input(_));
  main.variable(observer("viewof gradient")).define("viewof gradient", ["glsl"], _gradient);
  main.variable(observer("gradient")).define("gradient", ["Generators", "viewof gradient"], (G, _) => G.input(_));
  main.variable(observer()).define(["md"], _16);
  main.variable(observer("viewof gl")).define("viewof gl", ["DOM","width"], _gl);
  main.variable(observer("gl")).define("gl", ["Generators", "viewof gl"], (G, _) => G.input(_));
  main.variable(observer("render")).define("render", ["gl","webgl","model","now","textures"], _render);
  main.variable(observer("model")).define("model", ["webgl","fade","perm","grad","pnoise","norm","fbm","disp","wrap","gradient"], _model);
  main.variable(observer()).define(["md"], _20);
  main.variable(observer("glsl")).define("glsl", ["html","md"], _glsl);
  main.variable(observer()).define(["md"], _22);
  main.variable(observer("textures")).define("textures", ["webgl","gl"], _textures);
  main.variable(observer("webgl")).define("webgl", ["gl"], _webgl);
  main.variable(observer()).define(["md"], _25);
  main.variable(observer()).define(["md"], _26);
  return main;
}
