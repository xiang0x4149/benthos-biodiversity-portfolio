"""公开包检查：文件类型、链接、样本泄露模式、历史引用与 Git 对象。

这是一项可重复的发布检查，不宣称替代对所有可能敏感信息的人工审阅。
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import json
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
VOID={'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}


class Parser(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[];self.stack=[];self.errors=[]
    def handle_starttag(self,t,attrs):
        d=dict(attrs)
        if t not in VOID:self.stack.append(t)
        if 'id' in d:self.ids.append(d['id'])
        for k in ('href','src'):
            if k in d:self.links.append(d[k])
    def handle_startendtag(self,t,a):pass
    def handle_endtag(self,t):
        if not self.stack or self.stack[-1]!=t:self.errors.append('标签嵌套异常：'+t)
        else:self.stack.pop()


def scan_text(text, label):
    errors=[]
    patterns=[(r'(?<![A-Za-z])(?:Au|Su)?(?:GX|LHS|YH|YJ)[1-3](?!\d)','真实样点标识'),
              (r'/home/[^\s<>"\x27]+','本机绝对路径'),
              (r'(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,})','疑似访问凭证'),
              (r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----','私钥')]
    for pattern,reason in patterns:
        if re.search(pattern,text):errors.append(label+'：'+reason)
    return errors


def main():
    errors=[];checked=0;html_count=0;svg_count=0
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('.git','.venv','__pycache__')):continue
        rel=str(p.relative_to(ROOT));checked+=1
        if p.suffix.lower() in {'.pdf','.zip','.tar','.gz','.fasta','.rds','.rdata','.png','.jpg'}:errors.append(rel+'：未批准的二进制／原始资料类型')
        if p.is_symlink():errors.append(rel+'：发布包不得通过符号链接带入外部文件')
        if p.suffix=='.xlsx':
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith('.xml'):errors.extend(scan_text(z.read(n).decode(),rel+' 内部 XML'))
            continue
        text=p.read_text(encoding='utf-8')
        # 检查器中的拒绝模式本身不是泄露内容。
        if p.name!='check_release.py':errors.extend(scan_text(text,rel))
        if p.suffix=='.html':
            c=Parser();c.feed(text);c.close();html_count+=1
            errors.extend([rel+'：'+x for x in c.errors])
            if c.stack:errors.append(rel+'：存在未闭合标签')
            if len(c.ids)!=len(set(c.ids)):errors.append(rel+'：重复锚点')
            for link in c.links:
                u=urlsplit(link)
                if u.scheme:continue
                if u.path and not (p.parent/unquote(u.path)).exists():errors.append(rel+'：本地链接不存在 '+link)
                if not u.path and u.fragment and u.fragment not in c.ids:errors.append(rel+'：锚点不存在 '+link)
            for s in re.findall(r'<svg\b.*?</svg>',text,re.S):ET.fromstring(s)
        if p.suffix=='.svg':ET.fromstring(text);svg_count+=1
        if p.suffix=='.md':
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
                u=urlsplit(link)
                if not u.scheme and u.path and not (p.parent/unquote(u.path)).exists():
                    errors.append(rel+'：Markdown 本地链接不存在 '+link)
    if (ROOT/'.git').exists():
        selected=json.loads((ROOT/'provenance/lineage.json').read_text())['selected_commit']
        if subprocess.run(['git','cat-file','-e',selected+'^{commit}'],cwd=ROOT,capture_output=True).returncode==0:
            errors.append('发现私有项目来源提交对象；公开仓库不应导入它')
        checker_oid=subprocess.check_output(['git','hash-object','tools/check_release.py'],cwd=ROOT,text=True).strip()
        inventory=subprocess.check_output(['git','cat-file','--batch-all-objects','--batch-check=%(objectname) %(objecttype)'],cwd=ROOT,text=True)
        for line in inventory.splitlines():
            oid,kind=line.split()
            if kind=='blob':
                raw=subprocess.check_output(['git','cat-file','blob',oid],cwd=ROOT)
                if oid==checker_oid:continue
                if raw.startswith(b'PK'):
                    import io
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        for n in z.namelist():
                            if n.endswith('.xml'):errors.extend(scan_text(z.read(n).decode(),'Git 对象中的 XLSX'))
                else:errors.extend(scan_text(raw.decode('utf-8'),'Git 历史对象'))
    print(f'检查范围：{checked} 个发布文件；{html_count} 个 HTML；{svg_count} 个外部 SVG。')
    if errors:
        for x in errors:print('需处理：'+x)
        raise SystemExit(1)
    print('通过：本地链接、锚点、HTML 结构、SVG、敏感模式和已存在的初始 Git 历史对象检查。')


if __name__=='__main__':main()
