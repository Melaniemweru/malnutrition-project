print("📦 Starting run.py")  # Debug 1

try:
    from app import create_app
    print("✅ Successfully imported create_app")  # Debug 2

    app = create_app()
    print("✅ App created")  # Debug 3

    if __name__ == '__main__':
        print("🚀 Launching Flask app...")  # Debug 4
        app.run(debug=True, host='0.0.0.0')
except Exception as e:
    print("❌ ERROR in run.py:", str(e))  # Final catch-all




