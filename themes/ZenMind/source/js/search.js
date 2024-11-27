document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');

    // 确认元素加载成功
    if (!searchInput || !searchResults) {
        console.error('Search input or results container not found!');
        return;
    }

    // 搜索逻辑
    searchInput.addEventListener('input', function () {
        const query = searchInput.value.trim().toLowerCase(); // 转小写以匹配
        if (!query) {
            searchResults.style.display = 'none';
            searchResults.innerHTML = ''; // 清空结果
            return;
        }

        // 加载搜索 JSON 数据
       fetch('/search.json')
           .then(response => response.json())
           .then(data => {
               console.log('Search data:', data); // 调试输出
               const results = data.filter(item => {
                   const title = item.title ? item.title.toLowerCase() : '';
                   const content = item.content ? item.content.toLowerCase() : '';
                   return title.includes(query) || content.includes(query);
               });
               console.log('Filtered results:', results); // 调试过滤结果

               if (results.length > 0) {
                   searchResults.innerHTML = results
                       .map(result => `<a href="${result.url}">${result.title}</a>`)
                       .join('<br>');
                   searchResults.style.display = 'block';
               } else {
                   searchResults.innerHTML = '<p>No results found</p>';
                   searchResults.style.display = 'block';
               }
           })
           .catch(error => {
               console.error('Error fetching search.json:', error);
               searchResults.innerHTML = '<p>Error loading search data</p>';
               searchResults.style.display = 'block';
           });
    });
});