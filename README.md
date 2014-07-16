Localization
=================

Two main directories, one for creating a database map (CreateDB) and one for localizing within the database (Localize).

There are two directories, photos and kpd. photos are for photos and kpd are for keypoints and descriptors stored in yaml files.
The filenames for these must follow the same convention: `"_x_y_z_anglex_angley_anglez_.yaml||.jpg"`

Any filename that does not start with the delimiter specified in the input file will be ignored.

To begin, either start with directory of (preferably) square photos or start with pcd point cloud.

   * Create three folders for storing descriptors (kpd), above/below images (bwimages), and average pixel sum images (gsimages)
   * These should be in a common folder. Link to them in the Input file specified.
   * For directory of images:
      * Edit _CreateDB/build/InputsImages.txt_ with links to directory. Be sure filenames follow convention above.
      * Run `./createDBWithImages InputsImages.txt`.
   * For Point Cloud (pcd):
      * Edit _CreateDB/build/InputsPC.txt_ with links to point cloud and rendering preferences.
      * Run `./createDBWithPC InputsPC.txt`.

The database has now been created. To find best matches:

   * Place jpg file to match against in _Localize/build_ and link to it in _Localize/build/Inputs.txt_. 
   * Make sure delimiter is the same as the one used to create the database.
   * Link to common folders of directories in the input file.
   * Run `./localization Inputs.txt`
   * Images are displayed randomly, but saved in order of similarity. Smaller similarities correspond to closer matches.

To change how similarity is computed, edit _Similarity.h_ under `compareIandFs()`.

By Alex Rich and John Allard at Harvey Mudd College