#  Project ParallelBatchRCM
#  https://www.tugraz.at/institute/icg/research/team-steinberger/
#
#  Copyright (C) 2021 Institute for Computer Graphics and Vision,
#                     Graz University of Technology
#
#  Author(s):  Daniel Mlakar - daniel.mlakar ( at ) icg.tugraz.at
#              Martin Winter - martin.winter ( at ) icg.tugraz.at
#              Mathias Parger - mathias.parger ( at ) icg.tugraz.at
#              Markus Steinberger - steinberger ( at ) icg.tugraz.at
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#

import fnmatch, os, shutil

from urllib.parse import urlparse
import urllib.request

import tarfile


def retrieveMats(urls, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for addr in urls:
        mat_path = os.path.join(dst_dir, os.path.basename(urlparse(addr).path))
        if not os.path.exists(mat_path):
            urllib.request.urlretrieve(addr, mat_path)
            

def uncompressMats(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for tar_file in fnmatch.filter(os.listdir(src_dir), "*.tar.gz"):
        tar = tarfile.open(os.path.join(src_dir, tar_file))
        for file in tar.getnames():
            if not os.path.exists(os.path.join(dst_dir, file)):
                tar.extractall(dst_dir)
                break

def flattenTree(src_dir, dst_dir, pattern):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for dirpath, dirs, files in os.walk(src_dir, topdown=True):
        for file in fnmatch.filter(files, pattern):
            dest_file_path = os.path.join(dst_dir, file)
            if not os.path.exists(dest_file_path):
                shutil.copy2(os.path.join(dirpath, file), dest_file_path)

def replaceLineEndings(dir, pattern):
    for file in fnmatch.filter(os.listdir(dir), pattern):
        file_path = os.path.join(dir, file)
        with open(file_path, 'rb') as open_file:
            content = open_file.read()

        content = content.replace(b'\r\n', b'\n')

        with open(file_path, 'wb') as open_file:
            open_file.write(content)

        content = None

def main():
    tmp_dir = "./tmp"
    matrix_dir = "./"
    
    mats = ["https://suitesparse-collection-website.herokuapp.com/MM/HB/bcspwr10.tar.gz"
           , "https://suitesparse-collection-website.herokuapp.com/MM/Pothen/bodyy4.tar.gz"
           , "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/benzene.tar.gz"
           , "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/ncvxqp3.tar.gz"
	]

    
    retrieveMats(mats, tmp_dir)
    uncompressMats(tmp_dir, tmp_dir)
    flattenTree(tmp_dir, matrix_dir, "*.mtx")
    replaceLineEndings(matrix_dir, "*.mtx")

    shutil.rmtree(tmp_dir)



if __name__ == "__main__":
    main()