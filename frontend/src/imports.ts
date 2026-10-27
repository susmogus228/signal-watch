/** Read a CSV header without splitting delimiters inside quoted column names. */
export function csvHeaders(text:string):string[]{
 const first=text.replace(/^\uFEFF/,'').split(/\r?\n/,1)[0]??'';
 const count=(delimiter:string)=>{let quoted=false,n=0;for(let i=0;i<first.length;i++){if(first[i]==='"'){if(quoted&&first[i+1]==='"')i++;else quoted=!quoted;}else if(!quoted&&first[i]===delimiter)n++;}return n;};
 const delimiter=count(';')>count(',')?';':',';let quoted=false,current='';const headers:string[]=[];
 for(let i=0;i<first.length;i++){const c=first[i];if(c==='"'){if(quoted&&first[i+1]==='"'){current+='"';i++;}else quoted=!quoted;}else if(c===delimiter&&!quoted){headers.push(current.trim());current='';}else current+=c;}
 if(quoted)throw new Error('The CSV header has an unclosed quoted column name.');headers.push(current.trim());
 if(headers.some(h=>!h))throw new Error('Every CSV column must have a non-empty header.');
 if(new Set(headers).size!==headers.length)throw new Error('CSV column headers must be unique.');return headers;
}
