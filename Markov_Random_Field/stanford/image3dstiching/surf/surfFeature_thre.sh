# *  This code was used in the following articles:
# *  [1] Learning 3-D Scene Structure from a Single Still Image, 
# *      Ashutosh Saxena, Min Sun, Andrew Y. Ng, 
# *      In ICCV workshop on 3D Representation for Recognition (3dRR-07), 2007.
# *      (best paper)
# *  [2] 3-D Reconstruction from Sparse Views using Monocular Vision, 
# *      Ashutosh Saxena, Min Sun, Andrew Y. Ng, 
# *      In ICCV workshop on Virtual Representations and Modeling 
# *      of Large-scale environments (VRML), 2007. 
# *  [3] 3-D Depth Reconstruction from a Single Still Image, 
# *      Ashutosh Saxena, Sung H. Chung, Andrew Y. Ng. 
# *      International Journal of Computer Vision (IJCV), Aug 2007. 
# *  [6] Learning Depth from Single Monocular Images, 
# *      Ashutosh Saxena, Sung H. Chung, Andrew Y. Ng. 
# *      In Neural Information Processing Systems (NIPS) 18, 2005.
# *
# *  These articles are available at:
# *  http://make3d.stanford.edu/publications
# * 
# *  We request that you cite the papers [1], [3] and [6] in any of
# *  your reports that uses this code. 
# *  Further, if you use the code in image3dstiching/ (multiple image version),
# *  then please cite [2].
# *  
# *  If you use the code in third_party/, then PLEASE CITE and follow the
# *  LICENSE OF THE CORRESPONDING THIRD PARTY CODE.
# *
# *  Finally, this code is for non-commercial use only.  For further 
# *  information and to obtain a copy of the license, see 
# *
# *  http://make3d.stanford.edu/publications/code
# *
# *  Also, the software distributed under the License is distributed on an 
# * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# *  express or implied.   See the License for the specific language governing 
# *  permissions and limitations under the License.
# *
# */
#!/bin/bash

# Usage ./surfFeature_thre.sh <ImageName>
#
# Data:
#      ImageName.jpg in the dir/jpg/ folder
# 
# Create:
#  by surfFeature.sh
#  1) dir/pgm : lowest common denominator grayscale file format of the original *.jpg
#  2) dir/surf: surf features descriptor vecto       

dir=$1
img=$2
thres=$3
Type=$4

echo "Creating surf features for image in $dir$img"
echo $thres

if [ "$Type" == "_" ]; then
	Type="";
	echo 'Default Type';
fi

mkdir -p $dir/pgm
mkdir -p $dir/surf

echo $img
convert $dir/jpg/$img.jpg $dir/pgm/$img.pgm

# -d doubles image size before computation,  -u for upright featuresa, 
# -e for using extended descriptor (SURF-128)
../third_party/SURF-V1.0.8/surf.ln -i $dir/pgm/$img.pgm -o $dir/surf/$img.surf$Type -d -u -e -thres $thres
#    ../../third_party/SURF-V1.0.8/surf.ln -i $dir/pgm/$img.pgm -o $dir/surf/$img.surf_64 -d -u