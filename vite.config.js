import { defineConfig } from 'vite';
import {recordingAPI} from './scripts/recording-api.mjs';
export default defineConfig({plugins:[{name:'local-recording',configureServer(server){server.middlewares.use((req,res,next)=>{recordingAPI(req,res).then(handled=>{if(!handled)next();}).catch(next);});}}],base:'./',worker:{format:'es'},build:{target:'es2022',chunkSizeWarningLimit:2000},server:{host:'127.0.0.1',port:4187,strictPort:true},preview:{host:'127.0.0.1',port:4187,strictPort:true}});
