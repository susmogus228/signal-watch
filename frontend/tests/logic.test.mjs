import {test} from 'node:test';
import assert from 'node:assert/strict';
import {csvHeaders} from '../src/imports.ts';
import {request,flattenMetrics,time} from '../src/api.ts';
test('CSV mapping preserves quoted delimiters and SKAB semicolon headers',()=>{
 assert.deepEqual(csvHeaders('\uFEFFdatetime;Pressure;Volume Flow RateRMS\n2020;1;2'),['datetime','Pressure','Volume Flow RateRMS']);
 assert.deepEqual(csvHeaders('timestamp,"pressure, bar","sensor ""A"""\n'),['timestamp','pressure, bar','sensor "A"']);
});
test('Ambiguous and malformed headers fail before local upload',()=>{
 assert.throws(()=>csvHeaders('timestamp,a,a'),/unique/);
 assert.throws(()=>csvHeaders('timestamp,,a'),/non-empty/);
 assert.throws(()=>csvHeaders('timestamp,"unfinished'),/unclosed/);
});
test('Unknown measurements stay unavailable while zero stays zero',()=>{
 assert.deepEqual(flattenMetrics({delay:null,errors:0}),[['delay','Unavailable'],['errors','0']]);
 assert.equal(time(0),'00:00');assert.equal(time(3661),'61:01');
});
test('API validation messages surface instead of returning fake successful data',async()=>{
 const previous=globalThis.fetch;globalThis.fetch=async()=>new Response(JSON.stringify({detail:'Timestamp order is invalid.'}),{status:422});
 try{await assert.rejects(()=>request('/api/uploads'),/Timestamp order is invalid/);}finally{globalThis.fetch=previous;}
});
