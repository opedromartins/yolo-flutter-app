import 'package:flutter/material.dart';
import 'camera_inference_screen.dart';
import 'single_image_screen.dart';
import 'benchmark_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('YOLO Demo')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset('assets/applogo.png', height: 120),
              const SizedBox(height: 48),
              _navButton(
                context,
                icon: Icons.videocam,
                label: 'Camera Inference',
                screen: const CameraInferenceScreen(),
              ),
              const SizedBox(height: 16),
              _navButton(
                context,
                icon: Icons.image,
                label: 'Single Image',
                screen: const SingleImageScreen(),
              ),
              const SizedBox(height: 16),
              _navButton(
                context,
                icon: Icons.speed,
                label: 'Benchmark',
                screen: const BenchmarkScreen(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _navButton(
    BuildContext context, {
    required IconData icon,
    required String label,
    required Widget screen,
  }) {
    return SizedBox(
      width: double.infinity,
      height: 56,
      child: ElevatedButton.icon(
        onPressed: () => Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => screen),
        ),
        icon: Icon(icon),
        label: Text(label, style: const TextStyle(fontSize: 18)),
      ),
    );
  }
}
