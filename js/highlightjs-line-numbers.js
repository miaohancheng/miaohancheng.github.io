((e,t)=>{var n,r="hljs-ln",o="hljs-ln-code",a="hljs-ln-n",i="data-line-number",l=/\r\n|\r|\n/g;function s(n){try{var r,o=t.querySelectorAll("code.hljs,code.nohighlight");for(r in o)!o.hasOwnProperty(r)||o[r].classList.contains("nohljsln")||c(o[r],n)}catch(n){e.console.error("LineNumbers error: ",n)}}function c(t,n){"object"==typeof t&&e.setTimeout((function(){t.innerHTML=d(t,n)}),0)}function d(e,t){t={singleLine:(e=>e.singleLine||!1)(t=t||{}),startFrom:((e,t)=>{var n=1;return isFinite(t.startFrom)&&(n=t.startFrom),t=((e,t)=>e.hasAttribute(t)?e.getAttribute(t):null)(e,"data-ln-start-from"),null!==t?(e=>e&&(e=Number(e),isFinite(e))?e:1)(t):n})(e,t)},function e(t){var n=t.childNodes;for(var r in n)n.hasOwnProperty(r)&&0<h((r=n[r]).textContent)&&(0<r.childNodes.length?e(r):u(r.parentNode))}(e),e=e.innerHTML;var n=t,l=f(e);if(""===l[l.length-1].trim()&&l.pop(),1<l.length||n.singleLine){for(var s="",c=0,d=l.length;c<d;c++)s+=m('<tr><td class="{0} {1}" {3}="{5}"><div class="{2}" {3}="{5}"></div></td><td class="{0} {4}" {3}="{5}">{6}</td></tr>',["hljs-ln-line","hljs-ln-numbers",a,i,o,c+n.startFrom,0<l[c].length?l[c]:" "]);return m('<table class="{0}">{1}</table>',[r,s])}return e}function u(e){var t=e.className;if(/hljs-/.test(t)){for(var n=f(e.innerHTML),r=0,o="";r<n.length;r++)o+=m('<span class="{0}">{1}</span>\n',[t,0<n[r].length?n[r]:" "]);e.innerHTML=o.trim()}}function f(e){return 0===e.length?[]:e.split(l)}function h(e){return(e.trim().match(l)||[]).length}function m(e,t){return e.replace(/\{(\d+)\}/g,(function(e,n){return void 0!==t[n]?t[n]:e}))}e.hljs?(e.hljs.initLineNumbersOnLoad=function(n){"interactive"===t.readyState||"complete"===t.readyState?s(n):e.addEventListener("DOMContentLoaded",(function(){s(n)}))},e.hljs.lineNumbersBlock=c,e.hljs.lineNumbersValue=function(e,t){var n;if("string"==typeof e)return(n=document.createElement("code")).innerHTML=e,d(n,t)},(n=t.createElement("style")).type="text/css",n.innerHTML=m(".{0}{border-collapse:collapse}.{0} td{padding:0}.{1}:before{content:attr({2})}",[r,a,i]),t.getElementsByTagName("head")[0].appendChild(n)):e.console.error("highlight.js not detected!"),document.addEventListener("copy",(function(e){var t=window.getSelection();(e=>{for(var t=e;t;){if(t.className&&-1!==t.className.indexOf("hljs-ln-code"))return 1;t=t.parentNode}})(t.anchorNode)&&(t=-1!==window.navigator.userAgent.indexOf("Edge")?function(e){for(var t=e.toString(),n=e.anchorNode;"TD"!==n.nodeName;)n=n.parentNode;for(var r=e.focusNode;"TD"!==r.nodeName;)r=r.parentNode;e=parseInt(n.dataset.lineNumber);var a=parseInt(r.dataset.lineNumber);if(e==a)return t;var l,s=n.textContent,c=r.textContent;for(a<e&&(l=e,e=a,a=l,l=s,s=c,c=l);0!==t.indexOf(s);)s=s.slice(1);for(;-1===t.lastIndexOf(c);)c=c.slice(0,-1);for(var d=s,u=(e=>{for(var t=e;"TABLE"!==t.nodeName;)t=t.parentNode;return t})(n),f=e+1;f<a;++f){var h=m('.{0}[{1}="{2}"]',[o,i,f]);d+="\n"+u.querySelector(h).textContent}return d+"\n"+c}(t):t.toString(),e.clipboardData.setData("text/plain",t),e.preventDefault())}))})(window,document);