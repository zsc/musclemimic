import assets from './generated-assets.json';
export async function loadAsset(name,base) {
 const item=assets[name],response=await fetch(new URL(item.path,base));
 if(!response.ok)throw Error(`${name}: HTTP ${response.status}`);
 const data=await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
 if(data.byteLength!==item.decodedBytes)throw Error(`${name}: incomplete asset`);
 return data;
}
export async function loadJSONAsset(name,base){return JSON.parse(new TextDecoder().decode(await loadAsset(name,base)));}
export function unpack(data){
 const size=new DataView(data).getUint32(0,true);
 return {header:JSON.parse(new TextDecoder().decode(new Uint8Array(data,4,size))),offset:4+Math.ceil(size/4)*4};
}
export async function loadGeometry(base){
 const data=await loadAsset('geometry',base);let {header,offset}=unpack(data);
 const meshes=header.meshes.map(m=>{
  const vertices=new Float32Array(data,offset,m.vertices);offset+=m.vertices*4;
  const faces=new Uint32Array(data,offset,m.faces);offset+=m.faces*4;
  return {vertices,faces};
 });
 return {geoms:header.geoms,meshes};
}
