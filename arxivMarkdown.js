javascript:(function arxivMarkdown() {
  /**
   * Bookmarklet that builds a markdown link for an arxiv paper abstract into the clipboard, e.g.:
   *   [Title](http://arxiv.org/abs/xxxx.xxxxx "Author 1, Author 2")
   */
  var title = document.querySelector('.title').innerText;
  var link = window.location.toString();
  var authors = Array.prototype.map.call(document.querySelectorAll('.authors a'), function(author) {
    return author.innerText;
  }).join(', ');
  var markdown = ['[', title, '](', link, ' "', authors, '")'].join('');
  var input = document.createElement('input');
  input.value = markdown;
  document.body.appendChild(input);
  input.select();
  document.execCommand('copy');
  document.body.removeChild(input);
})();
