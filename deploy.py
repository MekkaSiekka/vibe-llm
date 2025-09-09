#!/usr/bin/env python3
"""
Deployment script for Local LLM Service

Automates the deployment process and provides easy setup options.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


class LLMDeployer:
    """Deployment manager for Local LLM Service."""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.project_root = Path(__file__).parent
    
    def check_docker(self) -> bool:
        """Check if Docker is installed and running."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Docker found: {result.stdout.strip()}")
            
            # Check if Docker daemon is running
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                check=True
            )
            print("✅ Docker daemon is running")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker not found or not running")
            print("   Please install Docker Desktop and start it")
            return False
    
    def check_docker_compose(self) -> bool:
        """Check if Docker Compose is available."""
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Docker Compose found: {result.stdout.strip()}")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker Compose not found")
            print("   Please install Docker Compose")
            return False
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            "models_cache",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def build_images(self):
        """Build Docker images."""
        print("🔨 Building Docker images...")
        try:
            subprocess.run(
                ["docker-compose", "build"],
                cwd=self.project_root,
                check=True
            )
            print("✅ Docker images built successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build Docker images: {e}")
            return False
        return True
    
    def start_development(self):
        """Start development environment."""
        print("🚀 Starting development environment...")
        try:
            subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=self.project_root,
                check=True
            )
            print("✅ Development environment started")
            print("   API: http://localhost:8000")
            print("   WebSocket: ws://localhost:8000/ws")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start development environment: {e}")
            return False
        return True
    
    def start_production(self):
        """Start production environment."""
        print("🚀 Starting production environment...")
        try:
            subprocess.run(
                ["docker-compose", "--profile", "production", "up", "-d"],
                cwd=self.project_root,
                check=True
            )
            print("✅ Production environment started")
            print("   API: http://localhost:8000")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start production environment: {e}")
            return False
        return True
    
    def stop_services(self):
        """Stop all services."""
        print("🛑 Stopping services...")
        try:
            subprocess.run(
                ["docker-compose", "down"],
                cwd=self.project_root,
                check=True
            )
            print("✅ Services stopped")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to stop services: {e}")
            return False
        return True
    
    def clean_up(self):
        """Clean up containers and volumes."""
        print("🧹 Cleaning up...")
        try:
            subprocess.run(
                ["docker-compose", "down", "-v"],
                cwd=self.project_root,
                check=True
            )
            subprocess.run(
                ["docker", "system", "prune", "-f"],
                check=True
            )
            print("✅ Cleanup completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to cleanup: {e}")
            return False
        return True
    
    def test_service(self):
        """Test the service."""
        print("🧪 Testing service...")
        try:
            # Wait a bit for service to start
            import time
            time.sleep(10)
            
            # Run test script
            subprocess.run(
                [sys.executable, "test_service.py"],
                cwd=self.project_root,
                check=True
            )
            print("✅ Service tests passed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Service tests failed: {e}")
            return False
        return True
    
    def show_logs(self):
        """Show service logs."""
        print("📋 Showing service logs...")
        try:
            subprocess.run(
                ["docker-compose", "logs", "-f"],
                cwd=self.project_root
            )
        except KeyboardInterrupt:
            print("\n✅ Logs stopped")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to show logs: {e}")
    
    def deploy(self, environment: str = "development", test: bool = True):
        """Full deployment process."""
        print("🚀 Local LLM Service Deployment")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_docker():
            return False
        
        if not self.check_docker_compose():
            return False
        
        # Create directories
        self.create_directories()
        
        # Build images
        if not self.build_images():
            return False
        
        # Start services
        if environment == "production":
            if not self.start_production():
                return False
        else:
            if not self.start_development():
                return False
        
        # Test service
        if test:
            if not self.test_service():
                return False
        
        print("\n🎉 Deployment completed successfully!")
        print("\nNext steps:")
        print("1. Test the API: curl http://localhost:8000/health")
        print("2. Try the client: python client_example.py")
        print("3. View logs: docker-compose logs -f")
        print("4. Stop services: docker-compose down")
        
        return True


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Local LLM Service")
    parser.add_argument(
        "command",
        choices=["deploy", "start", "stop", "test", "logs", "clean"],
        help="Deployment command"
    )
    parser.add_argument(
        "--env",
        choices=["development", "production"],
        default="development",
        help="Environment to deploy"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip testing after deployment"
    )
    
    args = parser.parse_args()
    
    deployer = LLMDeployer()
    
    if args.command == "deploy":
        success = deployer.deploy(
            environment=args.env,
            test=not args.no_test
        )
        sys.exit(0 if success else 1)
    
    elif args.command == "start":
        if args.env == "production":
            success = deployer.start_production()
        else:
            success = deployer.start_development()
        sys.exit(0 if success else 1)
    
    elif args.command == "stop":
        success = deployer.stop_services()
        sys.exit(0 if success else 1)
    
    elif args.command == "test":
        success = deployer.test_service()
        sys.exit(0 if success else 1)
    
    elif args.command == "logs":
        deployer.show_logs()
    
    elif args.command == "clean":
        success = deployer.clean_up()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

