UI Guide
========

This guide provides a complete walkthrough of the 3D Scene Reconstruction web application.
It is written for **non-technical users** and explains every screen, panel, button, and
indicator in plain language.

----

Overview of the Interface
--------------------------

When you open the application at http://localhost:5173, you are presented with a
**dashboard** divided into four main areas:

.. code-block:: text

   ┌──────────────────────────── HEADER ─────────────────────────────────┐
   │  App title · Current job ID · Pipeline stage indicator              │
   ├──────────────────────────┬──────────────────────────────────────────┤
   │       LEFT PANEL         │              RIGHT PANEL                 │
   │                          │                                          │
   │  · Upload Panel          │        3D Model Viewer                   │
   │  · Stage Tracker         │        (interactive point cloud)         │
   │  · Action Buttons        │                                          │
   ├──────────────────────────┴──────────────────────────────────────────┤
   │                    STATS TABLE (bottom)                             │
   │  Registration rate · Points · Drift status · Cluster breakdown      │
   └─────────────────────────────────────────────────────────────────────┘

----

Step 1 — Logging In
--------------------

Before you can upload data or view results, you must log in.

**What you see**

A centered login form with two fields: **Username** and **Password**.

**What to do**

1. Enter your username (default: ``admin``).
2. Enter your password (default: ``admin``).
3. Click **Login**.

**What happens internally**

The application sends your credentials to ``POST /auth/token``. On success, it
receives a **JWT access token** valid for 15 minutes, which is stored in your
browser's local storage. All subsequent API calls include this token in the
``Authorization`` header automatically.

**Error cases**

- *Invalid credentials* — you will see a red error message. Double-check your
  username and password. Contact your administrator if you do not have credentials.
- *Network error* — ensure the backend is running and reachable at the address shown
  in your browser.

**Session expiry**

After 15 minutes of inactivity, your token expires. The page will automatically
redirect you back to the login screen. Simply log in again to continue.

----

Step 2 — The Header
--------------------

The **Header** bar spans the full width of the page and shows:

- **Application name** — "Scene Reconstruction" branding on the left.
- **Job ID** — once you start a reconstruction, the active job's unique identifier
  (a UUID such as ``3f7a91b2-...``) is shown here. You can copy it to track the
  job via the API.
- **Stage pill** — a colour-coded indicator showing the current pipeline stage.
  It updates in real time as processing progresses.

----

Step 3 — The Upload Panel (Left Panel, Top)
--------------------------------------------

This is where you submit images for reconstruction.

**What you see**

A drag-and-drop zone with a **Choose File** button, and two optional text fields:
**Dataset Name** and **Scene Name**.

**What to do**

1. Click **Choose File** or drag a ZIP archive directly onto the upload area.
2. Optionally edit the **Dataset Name** (default: ``custom``) and **Scene Name**
   (default: ``scene_01``). These are used to organise your results internally
   and appear in the download filenames.
3. Click **Upload & Reconstruct**.

**What is a valid ZIP file?**

Your ZIP archive must contain image files with one of the following extensions:
``.jpg``, ``.jpeg``, ``.png``, ``.tif``, ``.tiff``, ``.bmp``, ``.webp``.
Images can be placed directly in the root of the ZIP or in subdirectories —
the system will find them automatically.

.. note::
   Maximum upload size is **500 MB** by default. If your dataset is larger,
   contact your administrator to increase the ``SCENE3D_MAX_UPLOAD_MB`` limit.

**What happens internally after you click Upload**

1. Your browser sends the ZIP file to ``POST /upload`` on the API.
2. The API immediately performs a **data drift check** — it computes brightness,
   contrast, sharpness, and resolution statistics for your images and compares
   them to the training baseline. If significant drift is detected, the system
   logs a warning and may automatically trigger retraining.
3. A **job record** is created with a unique Job ID.
4. The reconstruction pipeline is queued as a background task.
5. You are immediately returned the Job ID and the UI begins polling for status.

**While uploading is in progress**

The Upload button becomes greyed out and shows a spinner. You cannot start a
second job until the current one completes.

**Error cases**

- *"Upload exceeds 500 MB"* — reduce your dataset size or ask your administrator
  to raise the limit.
- *"Invalid ZIP"* — the file you selected is not a valid ZIP archive. Recreate
  it using your operating system's built-in zip tool.
- *"ZIP contains no supported image files"* — check that your images have a
  supported extension and are not hidden inside non-standard nested directories.

----

Step 4 — The Stage Tracker (Left Panel, Middle)
------------------------------------------------

Once a job is submitted, the **Stage Tracker** appears below the Upload Panel.
It shows you exactly where the pipeline is in the reconstruction process.

**The five pipeline stages**

.. list-table::
   :widths: 10 25 65
   :header-rows: 1

   * - #
     - Stage Name
     - What is happening
   * - 1
     - **Extracting** (10%)
     - Your ZIP file is unpacked on the server and images are saved to a temporary
       workspace. Image files are renamed to avoid collisions.
   * - 2
     - **Matching** (30%)
     - This is the most computationally expensive step. The MASt3R neural network
       runs on the GPU and finds correspondences (matching keypoints) between every
       pair of images. ALIKED and SuperPoint local feature detectors also run here
       to provide complementary sparse keypoints.
   * - 3
     - **Triangulating** (70%)
     - COLMAP's incremental Structure-from-Motion (SfM) mapper takes the keypoint
       matches and geometrically triangulates 3D points. Camera poses (rotation R
       and translation t) are determined for each registered image.
   * - 4
     - **Decimating** (85%)
     - The raw point cloud produced by COLMAP can contain millions of points. This
       stage applies voxel downsampling to reduce it to at most 500,000 points,
       making the result fast to download and render in the browser.
   * - 5
     - **Success** (100%)
     - Reconstruction is complete. The 3D viewer loads your point cloud automatically.

**Visual indicators**

Each stage is displayed as a row with:

- A **status icon** — grey circle (pending), spinning loader (active), green check
  (complete), or red X (failed).
- A **stage name**.
- A **progress bar** showing overall completion as a percentage.
- A **text message** describing what is happening in more detail.

**Timing expectations**

Reconstruction time depends on the number of images:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Dataset Size
     - Approximate Time
   * - < 20 images
     - 1–3 minutes
   * - 20–100 images
     - 3–10 minutes
   * - 100–500 images
     - 10–30 minutes
   * - > 500 images
     - 30–90 minutes

----

Step 5 — The 3D Model Viewer (Right Panel)
-------------------------------------------

Once reconstruction is complete, the interactive **3D Model Viewer** appears on the
right side of the dashboard.

**Controls**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Action
     - Effect
   * - **Left-click + drag**
     - Rotate the point cloud around its centre
   * - **Right-click + drag**
     - Pan (translate) the view
   * - **Scroll wheel**
     - Zoom in or out
   * - **Double-click**
     - Reset camera to the default viewpoint

**What you are looking at**

Each coloured dot in the viewer represents a **3D point** triangulated from two or
more images. Clusters of points naturally form the surfaces, edges, and textures of
the photographed scene. Camera positions are represented as small axis frames if
enabled.

**Cluster selector**

If the reconstruction produced multiple clusters (groups of images that the
algorithm determined were separate sub-scenes), a **cluster selector** appears above
the viewer. Click a cluster name to highlight only the points belonging to that group.

**Point density**

The point cloud displayed is the *decimated* version (maximum 500,000 points). The
original full-density cloud is available for download as a PLY file.

----

Step 6 — The Stats Table (Bottom Panel)
-----------------------------------------

Below the main panels, a **Statistics Table** summarises the reconstruction outcome.

**Columns**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Column
     - Meaning
   * - **Total Images**
     - Number of image files found in your ZIP archive.
   * - **Registered Images**
     - Number of images successfully placed in the 3D model. An image is
       "registered" when COLMAP was able to determine its pose.
   * - **Registration Rate**
     - Registered ÷ Total, expressed as a percentage. Higher is better.
       A rate above 90% is considered excellent.
   * - **Total 3D Points**
     - Total number of 3D points in the decimated point cloud.
   * - **Drift Detected**
     - Whether the uploaded images differ significantly from the training
       data distribution. ``No`` is ideal; ``Yes`` means the model may
       perform sub-optimally.
   * - **Drift Severity**
     - ``low``, ``moderate``, or ``high``. High severity may trigger
       automatic retraining.

**Cluster breakdown**

Below the summary row, each reconstructed cluster is listed with its own point count
and a download link for the individual PLY file.

----

Step 7 — Downloading Results
------------------------------

When the job status is **success**, two download options appear:

**Download Point Cloud (ZIP)**

Click the **Download .ply** button or use the download URL shown in the stats table.
This downloads a ZIP archive containing one PLY file per cluster. PLY files can be
opened in:

- **MeshLab** (free, recommended)
- **CloudCompare** (free)
- **Blender** (free, via the Point Cloud Visualiser add-on)
- **Open3D** (Python library)

**Download Submission CSV**

Click **Download CSV** to get the raw pose data in Kaggle IMC2025 submission format.
Each row contains the dataset name, scene name, image name, rotation matrix (9
semicolon-separated values), and translation vector (3 semicolon-separated values).

Images that could not be registered have ``nan`` values for their rotation and
translation.

----

Step 8 — Starting a New Reconstruction
----------------------------------------

After a job finishes (successfully or with failure), click **New Reconstruction**
in the left panel. This clears the current job and returns the Upload Panel to
its initial state.

----

Drift Warning Banner
---------------------

If significant data drift is detected during upload, a **yellow warning banner**
appears at the top of the stage tracker. It shows:

- Drift severity (``low``, ``moderate``, or ``high``)
- Which image statistics drifted (e.g., brightness, contrast, sharpness)

For ``high`` severity drift, the system automatically triggers a retraining job in
Airflow. You will still see your reconstruction results, but they may be less
accurate than usual.

----

Error States
-------------

**Job Failed**

If the pipeline encounters an unrecoverable error, the Stage Tracker shows a red
**Failed** indicator with an error message. Common causes are:

- Images are too blurry or lack sufficient overlap for matching.
- The ZIP contained fewer than 3 usable images (the minimum for reconstruction).
- The GPU ran out of memory (VRAM). Try reducing the number of images or contact
  your administrator.

**Session Expired**

If your JWT token expires mid-session, the application automatically detects the
401 response from the API and redirects you to the login page. Your job continues
running on the server — log back in and the job status will still be available
by entering the Job ID in the URL.

**Network Error**

If the API is unreachable, a banner message appears at the top of the page. Check
that the backend is running (``docker compose ps``) and that you are on the correct
network.

----

Frequently Asked Questions (UI)
---------------------------------

**Why are some of my images not in the point cloud?**
  Images are excluded when COLMAP cannot find enough keypoint matches connecting them
  to the rest of the scene. This typically happens when an image is blurry, captured
  at a very different scale, or has too little overlap with neighbouring images.
  The registration rate in the Stats Table tells you how many images were placed.

**Can I upload multiple scenes at once?**
  Currently, each upload creates a single job processing one scene. To process
  multiple scenes, upload them one at a time.

**The 3D viewer is blank after the job finishes. What should I do?**
  This means the reconstruction produced no valid 3D points. This can happen when
  there are fewer than 3 images with sufficient overlap, or when all images are
  from a single flat surface. Try adding more images with different viewpoints.

**How long are my results stored?**
  Results are stored in the server's ``/app/results/`` directory. They persist
  until the container is restarted or the results directory is manually cleared.
  Download your PLY and CSV files promptly after reconstruction.

**What does "drift detected" mean?**
  It means the statistical properties of your uploaded images (brightness, contrast,
  sharpness, resolution) differ from the images used to train the underlying models.
  The system can still reconstruct your scene, but accuracy may be lower. High-drift
  datasets may trigger an automatic retraining job overnight.
