(function () {
    const SEARCH_PATH = '/search.json';
    const RESULT_LIMIT = 8;
    const DEBOUNCE_DELAY = 200;
    let searchDataPromise = window.__zenmindSearchDataPromise || null;

    function debounce(fn, delay) {
        let timerId = null;
        return function (...args) {
            window.clearTimeout(timerId);
            timerId = window.setTimeout(() => fn.apply(this, args), delay);
        };
    }

    function stripHtml(html) {
        return (html || '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    }

    function excerpt(text) {
        return text.length > 140 ? `${text.slice(0, 137)}...` : text;
    }

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatDate(rawDate) {
        if (!rawDate) return '';
        const parsed = new Date(rawDate);
        if (Number.isNaN(parsed.getTime())) return '';
        return parsed.toISOString().slice(0, 10);
    }

    function getSearchData() {
        if (!searchDataPromise) {
            searchDataPromise = fetch(SEARCH_PATH)
                .then((response) => {
                    if (!response.ok) {
                        throw new Error('Failed to load search index.');
                    }
                    return response.json();
                })
                .then((data) => {
                    if (!Array.isArray(data)) return [];
                    return data.map((item) => {
                        const cleanContent = stripHtml(item.content || '');
                        return {
                            ...item,
                            _title: String(item.title || '').toLowerCase(),
                            _content: cleanContent.toLowerCase(),
                            _excerpt: excerpt(cleanContent)
                        };
                    });
                });
            window.__zenmindSearchDataPromise = searchDataPromise;
        }

        return searchDataPromise;
    }

    function renderResults(results, container) {
        if (results.length === 0) {
            container.innerHTML = '<p class="search-empty">No matching posts found.</p>';
            container.hidden = false;
            return;
        }

        container.innerHTML = results.map((item) => {
            const date = formatDate(item.date);
            const meta = date ? `<span class="search-result-meta">${escapeHtml(date)}</span>` : '';
            return `
                <a class="search-result" href="${escapeHtml(item.url)}">
                    <span class="search-result-title">${escapeHtml(item.title || 'Untitled')}</span>
                    ${meta}
                    <span class="search-result-excerpt">${escapeHtml(item._excerpt)}</span>
                </a>
            `;
        }).join('');
        container.hidden = false;
    }

    function scoreItem(item, query) {
        let score = 0;
        if (item._title.includes(query)) score += 3;
        if (item._content.includes(query)) score += 1;
        return score;
    }

    document.addEventListener('DOMContentLoaded', function () {
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');

        if (!searchInput || !searchResults || searchInput.dataset.searchBound === 'true') {
            return;
        }

        searchInput.dataset.searchBound = 'true';

        const handleSearch = debounce(async function () {
            const query = searchInput.value.trim().toLowerCase();

            if (!query) {
                searchResults.innerHTML = '';
                searchResults.hidden = true;
                return;
            }

            try {
                const data = await getSearchData();
                const results = data
                    .map((item) => ({
                        item,
                        score: scoreItem(item, query)
                    }))
                    .filter((entry) => entry.score > 0)
                    .sort((left, right) => {
                        if (right.score !== left.score) {
                            return right.score - left.score;
                        }
                        return new Date(right.item.date || 0).getTime() - new Date(left.item.date || 0).getTime();
                    })
                    .slice(0, RESULT_LIMIT)
                    .map((entry) => entry.item);

                renderResults(results, searchResults);
            } catch (error) {
                searchResults.innerHTML = '<p class="search-empty">Search data failed to load.</p>';
                searchResults.hidden = false;
            }
        }, DEBOUNCE_DELAY);

        searchInput.addEventListener('input', handleSearch);
        searchInput.addEventListener('keydown', function (event) {
            if (event.key === 'Escape') {
                searchInput.value = '';
                searchResults.innerHTML = '';
                searchResults.hidden = true;
                searchInput.blur();
            }
        });
    });
})();
