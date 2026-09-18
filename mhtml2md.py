#!/usr/bin/env python3
"""지후(zhihu) 저장본(.mhtml) → 번역용 중국어 markdown + 이미지.

  python3 mhtml2md.py "unclassified/reed_blog/xxx - 知乎.mhtml" out/slug images/

출력: out/slug/source.zh.md (본문), out/slug/*.jpg|png|webp (본문에 쓰인 이미지만).
세 번째 인자는 md 안에 들어갈 이미지 경로 prefix. 번역 후 md는 content/ 로,
이미지는 해당 섹션의 images/ 로 옮긴다.

pandoc 필요: pandoc -f html -t gfm-raw_html --wrap=none out/slug/_body.html -o out/slug/source.zh.md
  (-t gfm 은 div 안쪽을 통째로 raw HTML로 뱉으므로 gfm-raw_html 이어야 한다)
"""
import email, re, sys, os, hashlib
from email import policy
from urllib.parse import urlparse

IMG_EXT = {'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp',
           'image/gif': 'gif', 'image/svg+xml': 'svg'}


def slice_div(html, start):
    """start 위치에서 열린 div를 짝이 맞는 </div> 까지 잘라낸다."""
    depth = 0
    for m in re.finditer(r'<(/?)div\b', html[start:]):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return html[start:html.index('>', start + m.end()) + 1]
    return html[start:]


def extract(path, outdir, imgprefix):
    msg = email.message_from_binary_file(open(path, 'rb'), policy=policy.default)
    ps = [p for p in msg.walk() if not p.is_multipart()]

    html_part = next(p for p in ps if p.get_content_type() == 'text/html')
    # charset 미선언이라 get_content() 쓰면 한자가 깨진다
    html = html_part.get_payload(decode=True).decode('utf-8', 'replace')

    t = re.search(r'<title>(.*?)</title>', html, re.S)
    title = re.sub(r'\s*-\s*知乎\s*$', '', t.group(1).strip()) if t else os.path.basename(path)
    art = slice_div(html, re.search(r'<div class="RichText ztext Post-RichText[^"]*"', html).start())

    os.makedirs(outdir, exist_ok=True)
    saved = {}
    for p in ps:
        ct = p.get_content_type()
        loc = p.get('Content-Location', '')
        if not ct.startswith('image/') or not loc or loc in saved:
            continue
        data = p.get_payload(decode=True)
        if not data or len(data) < 3000:   # 아바타·아이콘
            continue
        base = re.sub(r'[^0-9a-zA-Z_.-]', '_', os.path.basename(urlparse(loc).path))
        name = (os.path.splitext(base)[0] or hashlib.md5(loc.encode()).hexdigest()) + '.' + IMG_EXT.get(ct, 'png')
        open(os.path.join(outdir, name), 'wb').write(data)
        saved[loc] = name          # 같은 basename + 다른 query URL 이 있으므로 name 기준으로 겹친다

    used = set()

    def fix(m):
        src = re.search(r'src="([^"]+)"', m.group(0))
        if src and src.group(1) in saved:
            used.add(src.group(1))
            return '<img src="%s%s">' % (imgprefix, saved[src.group(1)])
        return ''

    art = re.sub(r'<img [^>]*>', fix, art)
    for n in set(saved.values()) - {saved[u] for u in used}:
        os.remove(os.path.join(outdir, n))

    # pandoc 넘기기 전에 지워야 하는 것들:
    #   svg  → pandoc이 base64 data URI 이미지로 바꿔서 본문을 오염시킨다
    #   zhida 링크 / 각주 위첨자 → 본문 전체가 각주 번호로 뒤덮인다
    art = re.sub(r'<svg\b.*?</svg>', '', art, flags=re.S)
    art = re.sub(r'<a [^>]*href="https://zhida\.zhihu\.com/[^"]*"[^>]*>(.*?)</a>', r'\1', art, flags=re.S)
    art = re.sub(r'<sup [^>]*data-text=.*?</sup>', '', art, flags=re.S)

    open(os.path.join(outdir, '_body.html'), 'w').write('<h1>%s</h1>\n%s' % (title, art))
    print(title, '| imgs:', len(used))


if __name__ == '__main__':
    extract(sys.argv[1], sys.argv[2], sys.argv[3])
