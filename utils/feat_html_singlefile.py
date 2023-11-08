from pathlib import Path
import os, re
import subprocess as sp
import pandas as pd
from bs4 import BeautifulSoup
import base64
import argparse
from functools import partial

import logging
log = logging.getLogger(__name__)


def parser(context):
    
    parser = argparse.ArgumentParser(
        description="Flattens FSL FEAT report htmls to single file for Flywheel uploads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parse inputs similarly to cli ingest bids function

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f"Path does not exist: <{path}>.")
        return Path(path).absolute()

    def _is_file(path, parser):
        """Ensure a given path exists and it is a file."""
        path = _path_exists(path, parser)
        if not path.is_file():
            raise parser.error(f"Path should point to a file (or symlink of file): <{path}>.")
        return path

    PathExists = partial(_path_exists, parser=parser)
    IsFile = partial(_is_file, parser=parser)

    ##########################
    #   Required Arguments   #
    ##########################
    parser.add_argument(
        "featfile",
        action="store",
        metavar="REPORT.HTML",
        type=PathExists,
        help="file path for feat directory report.html to upload"
    )

    args = parser.parse_args()
    
    # add all args to context
    args_dict = args.__dict__
    context.update(args_dict)
    

def update_hyperlinks(obj):
    # takes old hyperlink format and changes it to within page references
    
    filelist=[]; reftext=[]
    
    for a in obj.find_all('a'):
        if ".html" in a['href']:
            
            # generate a list of all referenced files (they need to be added to the document later)
            filelist.append(str(Path(a['href']).resolve()))
            reftext.append(a.string)
            
            # update reference method
            a['href']="#"+a['href']
            del(a['target'])
    
    files = pd.DataFrame({"files":filelist, "refs":reftext})
    
    return files
        
    
def url_can_be_converted_to_data(tag):
    return tag.name.lower() == "img" and tag.has_attr('src') and not re.match('^data:', tag['src'])


def update_image_refs(obj,parentPath,htmlpath):
    "update all image references to be local paths"
    
    for a in obj.find_all('a'):
        for img in a.find_all('img'):
            if img.has_attr("src") and any(ele in img["src"] for ele in [".png",".svg",".jpeg"]):
                
                if os.path.isabs(img["src"]):
                    path=img["src"]
                else:
                    path=os.path.join(htmlpath,img["src"])
                try:
                    with open(path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read())

                    img['src'] = "data:image/png;base64, " + encoded_string.decode('utf-8')
                except OSError as e:
                    log.warning("File not found: %s", path)

            

def cleanup_image_refs(html):
    "update any remaining image links"
    for link in html.findAll(url_can_be_converted_to_data):
        if "tsplot" in link['src']:
            with open(os.path.join("tsplot",link['src'].replace("file:","")), "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            link['src'] = "data:image/png;base64, " + encoded_string.decode('utf-8')
        else:
            try:
                with open(link['src'].replace("file:",""), "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
            except OSError as e:
                log.warning("file not found: %s", link['src'].replace("file:",""))

            link['src'] = "data:image/png;base64, " + encoded_string.decode('utf-8')
            
        
def execute_cmd(cmd, dryrun=False):
    log.info("\n %s", cmd)
    if not dryrun:
        terminal = sp.Popen(
            cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
        )
        stdout, stderr = terminal.communicate()
        log.info("\n %s", stdout)
        log.info("\n %s", stderr)

        return stdout
    
    
def main(featfile):
    featfile = Path(featfile)
    
    cwd=os.getcwd()
    os.chdir(featfile.parent)
    
    # ---- build "base header for html with main links" ---- #
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data=os.path.join(dir_path, "base.html")
    # load the base file...
    with open(data) as inf:
        txt = inf.read()
    soup = BeautifulSoup(txt, 'html.parser')
    
    # load the main report file...
    with open(featfile) as inf:
        txt = inf.read()
    html1 = BeautifulSoup(txt, 'html.parser')
    
    df = update_hyperlinks(html1)
    files = df["files"]
    
    # add inital report "table" to base, then look through all subsequent files
    new_div = soup.new_tag("div",id="summary")
    new_return_link=soup.new_tag("a",href="#summary")
    new_return_link.string = "Return to Top"
    
    # add new contents inside division
    new_div.insert(0, html1.table)
    soup.body.insert(0, new_div)
    
    # ----- End Base Header ----- #
    
    allfiles = []; allrefs=[];
    for idx, f in enumerate(files):
        # load the main report file...
        htmlpath=Path(os.path.join(featfile.parent,f))
        
        with open(htmlpath) as inf:
            txt = inf.read()
        ihtml = BeautifulSoup(txt, 'html.parser')
        
        if not ihtml.find_all('body'):
            continue
            
        for tmp in ihtml.body.find_all('object'):
            tmp.decompose()
        
        df = update_hyperlinks(ihtml)
        df = df.drop_duplicates(subset=['files'])
        
        if any(name in f for name in ["firstlevel","reg"]):
            pass
        else:
            log.info(f)
            allfiles.extend(df['files'])
            allrefs.extend(df['refs'])

        update_image_refs(ihtml,featfile.parent,htmlpath.parent)
        
        # add inital report "table" to base, then look through all subsequent files
        new_div = soup.new_tag("div",id=Path(f).name)
        
        new_return_link=soup.new_tag("a",href="#summary")
        new_return_link.string = "Return to Top"
        
        # add new contents inside division
        ihtml.body.insert(0, new_return_link)
        new_div.insert(0, ihtml.body)
        soup.body.insert(idx+1, new_div)
        
    # ---- Loop through all secondary linked htmls ----- #
    for idxx, f in enumerate(allfiles):

        htmlpath=Path(f)
        with open(htmlpath) as inf:
            txt = inf.read()
        ihtml = BeautifulSoup(txt, 'html.parser')
        
        if not ihtml.find_all('body'):
            continue
            
        for tmp in ihtml.body.find_all('object'):
            tmp.decompose()
        
        ifiles, reftext = update_hyperlinks(ihtml)
        
        update_image_refs(ihtml,featfile.parent,htmlpath.parent)
        
        # add inital report "table" to base, then look through all subsequent files
        new_div = soup.new_tag("div",id=os.path.relpath(Path(f), start = featfile.parent))
        new_tag = soup.new_tag("h2")
        if allrefs[idxx]:
            new_tag.string = allrefs[idxx]
        else:
            new_tag.string = allfiles[idxx]
        
        new_return_link=soup.new_tag("a",href="#summary")
        new_return_link.string = "Return to Top"
        
        # add new contents inside division
        ihtml.body.insert(0, new_return_link)
        ihtml.body.insert(1, new_tag)
        new_div.insert(0, ihtml.body)
        soup.body.insert(idxx+idx+1, new_div)
    
        
    # ---- write output ------ #
    cleanup_image_refs(soup)
    
    log.info("Writing html: %s",os.path.join(featfile.parent,"index.html"))
    html = soup.prettify(formatter="html")
    
    with open(os.path.join(featfile.parent,"index.html"), "w") as outf:
        outf.write(str(html))

    os.chdir(cwd)
        

if __name__ == "__main__":
    

    pycontext = dict()
    
    parser(pycontext)
    main(pycontext["featfile"])
    
    