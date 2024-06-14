import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:gallery_saver/gallery_saver.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart' as path_provider;
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:assets_audio_player/assets_audio_player.dart';
import 'package:syncfusion_flutter_sliders/sliders.dart';
import 'package:flutter_tts/flutter_tts.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp(this.cameras, {super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Camera App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: PhotoScreen(cameras: cameras),
    );
  }
}

class CaptionScreen extends StatefulWidget {
  final String caption;

  const CaptionScreen({super.key, required this.caption});

  @override
  _CaptionScreenState createState() => _CaptionScreenState();
}

class _CaptionScreenState extends State<CaptionScreen> {
  late FlutterTts flutterTts;

  @override
  void initState() {
    super.initState();
    flutterTts = FlutterTts();
    _speak(widget.caption);
  }

  Future<void> _speak(String text) async {
    await flutterTts.setLanguage("en-US");
    await flutterTts.setSpeechRate(0.3);
    await flutterTts.speak(text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Generated Caption'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              widget.caption,
              style: TextStyle(fontSize: 44, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
          ),
          Spacer(),
          ElevatedButton(
            onPressed: () {
              _speak(widget.caption);
            },
            child: Text('Repeat',
                style: TextStyle(fontSize: 40, color: Colors.white)),
            style: ElevatedButton.styleFrom(
              backgroundColor: Color.fromARGB(255, 222, 106, 12),
              padding: EdgeInsets.symmetric(horizontal: 50, vertical: 50),
            ),
          ),
          SizedBox(height: 100),
          ElevatedButton(
            onPressed: () {
              _speak("Going back");
              Navigator.pop(context);
            },
            child: Text('  Back  ',
                style: TextStyle(fontSize: 40, color: Colors.white)),
            style: ElevatedButton.styleFrom(
              backgroundColor: Color.fromRGBO(27, 23, 105, 1),
              padding: EdgeInsets.symmetric(horizontal: 50, vertical: 50),
            ),
          ),
          SizedBox(height: 80),
        ],
      ),
    );
  }

  @override
  void dispose() {
    flutterTts.stop();
    super.dispose();
  }
}

class PhotoScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const PhotoScreen({super.key, required this.cameras});

  @override
  State<PhotoScreen> createState() => _PhotoScreenState();
}

class _PhotoScreenState extends State<PhotoScreen> {
  CameraController? _controller;
  bool isCapturing = false;
  int _selectedCameraIndex = 0;
  bool _isFrontCamera = false;
  bool _isFlashOn = false;
  Offset? _focusPoint;
  double _currentZoom = 1.0;
  File? _capturedImage;

  AssetsAudioPlayer audioPlayer = AssetsAudioPlayer();
  late FlutterTts flutterTts;

  @override
  void initState() {
    super.initState();
    flutterTts = FlutterTts();
    initializeCamera();
    _announce("Camera screen opened. You are on the photo screen.");
  }

  Future<void> _announce(String text) async {
    await flutterTts.setLanguage("en-US");
    await flutterTts.setSpeechRate(0.3);
    await flutterTts.speak(text);
  }

  void initializeCamera() async {
    try {
      List<CameraDescription> cameras = await availableCameras();
      if (cameras.isNotEmpty) {
        _controller = CameraController(cameras[0], ResolutionPreset.max);
        await _controller?.initialize();
        if (mounted) {
          setState(() {});
        }
        _announce("Camera initialized. Ready to take a photo.");
      } else {
        _announce('No cameras found');
      }
    } catch (e) {
      print('Error initializing camera: $e');
      _announce('Error initializing camera');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    flutterTts.stop();
    super.dispose();
  }

  void _toggleFlashLight() {
    if (_isFlashOn) {
      _controller?.setFlashMode(FlashMode.off);
      setState(() {
        _isFlashOn = false;
      });
      _announce("Flash turned off");
    } else {
      _controller?.setFlashMode(FlashMode.torch);
      setState(() {
        _isFlashOn = true;
      });
      _announce("Flash turned on");
    }
  }

  void _switchCamera() async {
    if (_controller != null) {
      await _controller!.dispose();
    }

    _selectedCameraIndex = (_selectedCameraIndex + 1) % widget.cameras.length;

    _initCamera(_selectedCameraIndex);
  }

  Future<void> _initCamera(int cameraIndex) async {
    _controller =
        CameraController(widget.cameras[cameraIndex], ResolutionPreset.max);

    try {
      await _controller!.initialize();
      setState(() {
        if (cameraIndex == 0) {
          _isFrontCamera = false;
        } else {
          _isFrontCamera = true;
        }
      });
      _announce(
          _isFrontCamera ? "Front camera activated" : "Rear camera activated");
    } catch (e) {
      print('Error initializing camera: $e');
      _announce('Error initializing camera');
    }
  }

  Future<String?> uploadImage(File imageFile) async {
    final uri = Uri.parse("http://10.111.230.187:5000/predict");
    final request = http.MultipartRequest('POST', uri);
    request.files
        .add(await http.MultipartFile.fromPath('file', imageFile.path));
    final response = await request.send();

    if (response.statusCode == 200) {
      final responseData = await response.stream.bytesToString();
      final jsonResponse = json.decode(responseData);
      return jsonResponse["caption"];
    } else {
      print("Failed to upload image");
      _announce("Failed to upload image");
      return null;
    }
  }

  void capturePhoto() async {
    if (!_controller!.value.isInitialized) {
      print('Error: Camera is not initialized');
      _announce('Error: Camera is not initialized');
      return;
    }

    final Directory appDir =
        await path_provider.getApplicationSupportDirectory();
    final String capturePath = path.join(appDir.path, '${DateTime.now()}.jpg');

    if (_controller!.value.isTakingPicture) {
      print('Camera is already capturing a picture');
      _announce('Camera is already capturing a picture');
      return;
    }

    try {
      setState(() {
        isCapturing = true;
      });

      final XFile capturedImage = await _controller!.takePicture();
      String imagePath = capturedImage.path;
      await GallerySaver.saveImage(imagePath);
      print("Photo captured and saved to gallery");
      _announce("Photo captured and saved to gallery");
      print("imagePath:" + imagePath);

      audioPlayer.open(Audio('music/camera-13695.mp3'));
      audioPlayer.play();

      setState(() {
        _capturedImage = File(imagePath);
      });

      String? caption = await uploadImage(_capturedImage!);
      if (caption != null) {
        print("Generated Caption: $caption");
        _announce("Generated Caption: $caption");
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => CaptionScreen(caption: caption),
          ),
        );
      } else {
        print("Caption is null");
        _announce("Caption is null");
      }
    } catch (e) {
      print("Error capturing photo: $e");
      _announce("Error capturing photo");
    } finally {
      setState(() {
        isCapturing = false;
      });
    }
  }

  void zoomCamera(double value) {
    _currentZoom = value;
    _controller?.setZoomLevel(value);
    _announce("Zoom level: ${value.toStringAsFixed(1)}");
  }

  Future<void> _setFocusPoint(Offset point) async {
    if (_controller != null && _controller!.value.isInitialized) {
      try {
        final double x = point.dx.clamp(0.0, 1.0);
        final double y = point.dy.clamp(0.0, 1.0);
        await _controller!.setFocusPoint(Offset(x, y));
        await _controller!.setFocusMode(FocusMode.auto);
        setState(() {
          _focusPoint = Offset(x, y);
        });

        await Future.delayed(const Duration(milliseconds: 2));
        setState(() {
          _focusPoint = null;
        });
        _announce("Focus set");
      } catch (e) {
        print("Error setting focus point: $e");
        _announce("Error setting focus point");
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(
        builder: (BuildContext context, BoxConstraints constraints) {
          if (_controller == null || !_controller!.value.isInitialized) {
            return Center(
              child: CircularProgressIndicator(),
            );
          }
          return Stack(
            children: [
              Positioned(
                top: 0,
                left: 0,
                right: 0,
                child: Container(
                  height: 50,
                  decoration: BoxDecoration(
                    color: Colors.black,
                  ),
                  // child: Row(
                  //   mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  //   children: [
                  //     Padding(
                  //       padding: const EdgeInsets.all(10.0),
                  //       child: GestureDetector(
                  //         onTap: () {
                  //           _toggleFlashLight();
                  //         },
                  //         child: _isFlashOn == false
                  //             ? Icon(Icons.flash_on, color: Colors.white)
                  //             : Icon(Icons.flash_off, color: Colors.white),
                  //       ),
                  //     ),
                  //   ],
                  // ),
                ),
              ),
              Positioned.fill(
                top: 50,
                bottom: _isFrontCamera == false ? 0 : 150,
                child: AspectRatio(
                  aspectRatio: _controller!.value.aspectRatio ?? 16 / 9,
                  // Null check for aspect ratio
                  child: GestureDetector(
                    onTapDown: (TapDownDetails details) {
                      final Offset tapPosition = details.localPosition;
                      final Offset relativeTapPossition = Offset(
                        tapPosition.dx / constraints.maxWidth,
                        tapPosition.dy / constraints.maxHeight,
                      );
                      _setFocusPoint(relativeTapPossition);
                    },
                    child: CameraPreview(_controller!),
                  ),
                ),
              ),
              Positioned(
                top: 50,
                right: 10, // Change from left to right
                child: SfSlider.vertical(
                  max: 5.0,
                  min: 1.0,
                  activeColor: Colors.white,
                  value: _currentZoom,
                  onChanged: (dynamic value) {
                    setState(() {
                      zoomCamera(value);
                    });
                  },
                ),
              ),
              Positioned(
                top: 50,
                left:
                    10, // Adjust the position to be on the right side of the zoom slider
                child: GestureDetector(
                  onTap: () {
                    _toggleFlashLight();
                  },
                  child: _isFlashOn == false
                      ? Icon(Icons.flash_on, color: Colors.white, size: 80)
                      : Icon(Icons.flash_off, color: Colors.white, size: 80),
                ),
              ),
              if (_focusPoint != null)
                Positioned.fill(
                  top: 50,
                  child: Align(
                    alignment: Alignment(
                        _focusPoint!.dx * 2 - 1, _focusPoint!.dy * 2 - 1),
                    child: Container(
                      width: 60,
                      height: 100,
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.white, width: 2),
                        shape: BoxShape.circle,
                      ),
                    ),
                  ),
                ),
              Positioned(
                bottom: 0,
                left: 0,
                right: 0,
                child: Container(
                  height: 200,
                  decoration: BoxDecoration(
                    color:
                        _isFrontCamera == false ? Colors.black45 : Colors.black,
                  ),
                  child: Column(
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(10.0),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            Expanded(
                              child: Center(
                                child: Text(
                                  "Photo",
                                  style: TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 50,
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Row(
                              children: [
                                Expanded(
                                  child: Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      _capturedImage != null
                                          ? Container(
                                              width: 60,
                                              height: 100,
                                              child: Image.file(
                                                _capturedImage!,
                                                fit: BoxFit.cover,
                                              ),
                                            )
                                          : Container(),
                                    ],
                                  ),
                                ),
                                Expanded(
                                  child: GestureDetector(
                                    onTap: () {
                                      capturePhoto();
                                    },
                                    child: Center(
                                      child: Container(
                                        height: 90,
                                        width: 150,
                                        decoration: BoxDecoration(
                                          color: Colors.transparent,
                                          borderRadius:
                                              BorderRadius.circular(50),
                                          border: Border.all(
                                            color: Colors.white,
                                            width: 4,
                                            style: BorderStyle.solid,
                                          ),
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                                Expanded(
                                  child: GestureDetector(
                                    onTap: () {
                                      _switchCamera();
                                    },
                                    child: Icon(
                                      Icons.cameraswitch,
                                      color: Colors.white,
                                      size: 90,
                                    ),
                                  ),
                                ),
                              ],
                            )
                          ],
                        ),
                      )
                    ],
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
