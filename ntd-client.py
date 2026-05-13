"""
Network Threat Detector Client - Fully Interactive

Features:
- Interactive menu system
- Auto-report generation
- Colorful UI with NIRVANA banner
- JWT authentication

Usage:
    python ntd-client.py                    # Interactive mode
    python ntd-client.py login              # Direct login
    python ntd-client.py analyze file.csv   # Direct analysis
"""

import requests
import json
import sys
import os
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
TOKEN_FILE = ".ntd_token"


class Colors:
    """ANSI color codes"""
    RED = '\033[91m'
    RED2 = '\033[38;5;196m'
    PINK = '\033[38;5;205m'
    LIGHT_PINK = '\033[38;5;219m'
    WHITE = '\033[97m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class ThreatDetectorClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.token = self.load_token()
        self.username = None
    
    def _print_banner(self):
        """Print colorful NIRVANA banner"""
        self._clear_screen()
        print()
        print()
        print(f"        {Colors.RED} ███╗   ██╗ ██╗ ██████╗  ██╗   ██╗  █████╗  ███╗   ██╗  █████╗ {Colors.RESET}")
        print(f"        {Colors.RED} ████╗  ██║ ██║ ██╔══██╗ ██║   ██║ ██╔══██╗ ████╗  ██║ ██╔══██╗{Colors.RESET}")
        print(f"        {Colors.RED2} ██╔██╗ ██║ ██║ ██████╔╝ ██║   ██║ ███████║ ██╔██╗ ██║ ███████║{Colors.RESET}")
        print(f"        {Colors.PINK} ██║╚██╗██║ ██║ ██╔══██╗ ╚██╗ ██╔╝ ██╔══██║ ██║╚██╗██║ ██╔══██║{Colors.RESET}")
        print(f"        {Colors.LIGHT_PINK} ██║ ╚████║ ██║ ██║  ██║  ╚████╔╝  ██║  ██║ ██║ ╚████║ ██║  ██║{Colors.RESET}")
        print(f"        {Colors.WHITE} ╚═╝  ╚═══╝ ╚═╝ ╚═╝  ╚═╝   ╚═══╝   ╚═╝  ╚═╝ ╚═╝  ╚═══╝ ╚═╝  ╚═╝{Colors.RESET}")
        print()
        print(f"                        {Colors.CYAN}local ai-threat detector{Colors.RESET}")
        print()
        print()
    
    def _clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _print_separator(self, char="=", length=60):
        """Print separator line"""
        print(f"{Colors.CYAN}{char * length}{Colors.RESET}")
    
    def _wait_for_enter(self):
        """Wait for user to press Enter"""
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")
    
    def load_token(self):
        """Load saved token from file"""
        try:
            with open(TOKEN_FILE, 'r') as f:
                data = json.load(f)
                self.username = data.get('username')
                return data.get('access_token')
        except:
            return None
    
    def save_token(self, token_data):
        """Save token to file"""
        with open(TOKEN_FILE, 'w') as f:
            json.dump(token_data, f)
    
    def get_headers(self):
        """Get authorization headers"""
        if not self.token:
            print(f"{Colors.RED}❌ Not logged in. Please login first.{Colors.RESET}")
            return None
        
        return {"Authorization": f"Bearer {self.token}"}
    
    def show_main_menu(self):
        """Show main interactive menu"""
        while True:
            self._print_banner()
            
            # Show login status
            if self.token:
                print(f"{Colors.GREEN}● Logged in as: {Colors.CYAN}{self.username}{Colors.RESET}")
            else:
                print(f"{Colors.RED}● Not logged in{Colors.RESET}")
            
            print()
            self._print_separator()
            print(f"{Colors.BOLD}{Colors.CYAN}MAIN MENU{Colors.RESET}")
            self._print_separator()
            print()
            
            if not self.token:
                print(f"{Colors.YELLOW}1.{Colors.RESET} Register New Account")
                print(f"{Colors.YELLOW}2.{Colors.RESET} Login")
                print(f"{Colors.YELLOW}3.{Colors.RESET} Check API Health")
                print(f"{Colors.YELLOW}0.{Colors.RESET} Exit")
            else:
                print(f"{Colors.YELLOW}1.{Colors.RESET} Analyze Network Traffic")
                print(f"{Colors.YELLOW}2.{Colors.RESET} List Available Models")
                print(f"{Colors.YELLOW}3.{Colors.RESET} View My Profile")
                print(f"{Colors.YELLOW}4.{Colors.RESET} Check API Health")
                print(f"{Colors.YELLOW}5.{Colors.RESET} Logout")
                print(f"{Colors.YELLOW}0.{Colors.RESET} Exit")
            
            print()
            choice = input(f"{Colors.CYAN}Select option: {Colors.RESET}").strip()
            
            if not self.token:
                if choice == "1":
                    self.register()
                elif choice == "2":
                    self.login()
                elif choice == "3":
                    self.health_check()
                    self._wait_for_enter()
                elif choice == "0":
                    print(f"\n{Colors.CYAN}Thank you for using NIRVANA! Goodbye!{Colors.RESET}\n")
                    sys.exit(0)
                else:
                    print(f"{Colors.RED}Invalid option!{Colors.RESET}")
                    self._wait_for_enter()
            else:
                if choice == "1":
                    self.interactive_analyze()
                elif choice == "2":
                    self.list_models()
                    self._wait_for_enter()
                elif choice == "3":
                    self.get_me()
                    self._wait_for_enter()
                elif choice == "4":
                    self.health_check()
                    self._wait_for_enter()
                elif choice == "5":
                    self.logout()
                    self._wait_for_enter()
                elif choice == "0":
                    print(f"\n{Colors.CYAN}Thank you for using NIRVANA! Goodbye!{Colors.RESET}\n")
                    sys.exit(0)
                else:
                    print(f"{Colors.RED}Invalid option!{Colors.RESET}")
                    self._wait_for_enter()
    
    def interactive_analyze(self):
        """Interactive analysis workflow"""
        self._print_banner()
        self._print_separator()
        print(f"{Colors.BOLD}{Colors.CYAN}THREAT ANALYSIS{Colors.RESET}")
        self._print_separator()
        print()
        
        # Get CSV file
        print(f"{Colors.CYAN}Enter path to CSV file:{Colors.RESET}")
        csv_file = input(f"{Colors.WHITE}> {Colors.RESET}").strip()
        
        if not csv_file:
            print(f"{Colors.RED}❌ No file specified!{Colors.RESET}")
            self._wait_for_enter()
            return
        
        file_path = Path(csv_file)
        if not file_path.exists():
            print(f"{Colors.RED}❌ File not found: {csv_file}{Colors.RESET}")
            self._wait_for_enter()
            return
        
        # Ask about LLM analysis
        print(f"\n{Colors.CYAN}Enable LLM analysis? (y/n) [default: y]:{Colors.RESET}")
        use_llm = input(f"{Colors.WHITE}> {Colors.RESET}").strip().lower() != 'n'
        
        llm_model = "gemma:1b"
        if use_llm:
            # Show model selection
            print(f"\n{Colors.CYAN}Select LLM model:{Colors.RESET}")
            print(f"  {Colors.YELLOW}1.{Colors.RESET} llama3.2:1b {Colors.GREEN}(recommended, fast){Colors.RESET}")
            print(f"  {Colors.YELLOW}2.{Colors.RESET} llama3.2:3b (detailed)")
            print(f"  {Colors.YELLOW}3.{Colors.RESET} phi3:mini (alternative)")
            print(f"  {Colors.YELLOW}4.{Colors.RESET} Gemma3:1b (alternative)")
            print(f"  {Colors.YELLOW}5.{Colors.RESET} Custom model")
            
            model_choice = input(f"\n{Colors.CYAN}Select model [1]: {Colors.RESET}").strip() or "1"
            
            if model_choice == "1":
                llm_model = "llama3.2:1b"
            elif model_choice == "2":
                llm_model = "llama3.2:3b"
            elif model_choice == "3":
                llm_model = "phi3:mini"
            elif model_choice == "4":
                llm_model = "Gemma3:1b"
            elif model_choice == "5":
                llm_model = input(f"{Colors.CYAN}Enter model name: {Colors.RESET}").strip()
        
        # Ask about report generation
        print(f"\n{Colors.CYAN}Generate Markdown report? (y/n) [default: y]:{Colors.RESET}")
        generate_report = input(f"{Colors.WHITE}> {Colors.RESET}").strip().lower() != 'n'
        
        # Confirm and analyze
        print(f"\n{Colors.CYAN}Analysis Configuration:{Colors.RESET}")
        print(f"  File: {Colors.WHITE}{csv_file}{Colors.RESET}")
        print(f"  LLM Analysis: {Colors.GREEN if use_llm else Colors.RED}{'Yes' if use_llm else 'No'}{Colors.RESET}")
        if use_llm:
            print(f"  LLM Model: {Colors.CYAN}{llm_model}{Colors.RESET}")
        print(f"  Generate Report: {Colors.GREEN if generate_report else Colors.RED}{'Yes' if generate_report else 'No'}{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}Proceed with analysis? (y/n) [default: y]:{Colors.RESET}")
        proceed = input(f"{Colors.WHITE}> {Colors.RESET}").strip().lower() != 'n'
        
        if proceed:
            print()
            self.analyze_csv(csv_file, use_llm=use_llm, llm_model=llm_model, 
                           generate_report=generate_report)
            self._wait_for_enter()
        else:
            print(f"{Colors.YELLOW}Analysis cancelled.{Colors.RESET}")
            self._wait_for_enter()
    
    def register(self):
        """Register new user"""
        self._print_banner()
        self._print_separator()
        print(f"{Colors.BOLD}{Colors.CYAN}USER REGISTRATION{Colors.RESET}")
        self._print_separator()
        print()
        
        username = input(f"{Colors.CYAN}Username: {Colors.RESET}").strip()
        
        import getpass
        password = getpass.getpass(f"{Colors.CYAN}Password: {Colors.RESET}")
        password_confirm = getpass.getpass(f"{Colors.CYAN}Confirm password: {Colors.RESET}")
        
        if password != password_confirm:
            print(f"\n{Colors.RED}❌ Passwords don't match{Colors.RESET}")
            self._wait_for_enter()
            return
        
        email = input(f"{Colors.CYAN}Email (optional): {Colors.RESET}").strip() or None
        
        try:
            response = requests.post(
                f"{self.base_url}/register",
                json={"username": username, "password": password, "email": email}
            )
            
            if response.status_code == 200:
                user = response.json()
                print(f"\n{Colors.GREEN}✅ Registration successful!{Colors.RESET}")
                print(f"   Username: {Colors.CYAN}{user['username']}{Colors.RESET}")
                print(f"   Role: {Colors.YELLOW}{user['role']}{Colors.RESET}")
            else:
                print(f"\n{Colors.RED}❌ {response.json()['detail']}{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
        
        self._wait_for_enter()
    
    def login(self, username=None, password=None):
        """Login"""
        if not username:
            self._print_banner()
            self._print_separator()
            print(f"{Colors.BOLD}{Colors.CYAN}USER LOGIN{Colors.RESET}")
            self._print_separator()
            print()
            username = input(f"{Colors.CYAN}Username: {Colors.RESET}").strip()
            import getpass
            password = getpass.getpass(f"{Colors.CYAN}Password: {Colors.RESET}")
        
        try:
            response = requests.post(
                f"{self.base_url}/login",
                json={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                data['username'] = username
                self.save_token(data)
                self.token = data['access_token']
                self.username = username
                print(f"\n{Colors.GREEN}✅ Login successful!{Colors.RESET}")
            else:
                print(f"\n{Colors.RED}❌ {response.json()['detail']}{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
        
        if not username:
            self._wait_for_enter()
    
    def logout(self):
        """Logout"""
        Path(TOKEN_FILE).unlink(missing_ok=True)
        self.token = None
        self.username = None
        print(f"\n{Colors.GREEN}✅ Logged out{Colors.RESET}")
    
    def get_me(self):
        """Get profile"""
        headers = self.get_headers()
        if not headers:
            return
        
        try:
            response = requests.get(f"{self.base_url}/me", headers=headers)
            if response.status_code == 200:
                user = response.json()
                print(f"\n{Colors.CYAN}USER PROFILE{Colors.RESET}\n")
                print(f"Username: {Colors.CYAN}{user['username']}{Colors.RESET}")
                print(f"Role: {Colors.YELLOW}{user['role']}{Colors.RESET}")
                print(f"Email: {user.get('email', 'N/A')}")
        except Exception as e:
            print(f"{Colors.RED}❌ {e}{Colors.RESET}")
    
    def analyze_csv(self, csv_file_path, use_llm=True, llm_model="Gemma3:1b", generate_report=True):
        """Analyze CSV"""
        headers = self.get_headers()
        if not headers:
            return
        
        try:
            import pandas as pd
            import numpy as np
            
            df = pd.read_csv(csv_file_path)
            print(f"{Colors.CYAN}📂 Loaded {len(df)} records{Colors.RESET}")
            
            df = df.fillna(0).replace([np.inf, -np.inf], 0)
            network_data = df.to_dict('records')
            
            response = requests.post(
                f"{self.base_url}/analyze",
                headers=headers,
                json={"network_data": network_data, "use_llm": use_llm, "llm_model": llm_model}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"{Colors.GREEN}✅ Analysis complete!{Colors.RESET}\n")
                print(f"Threats: {Colors.RED}{result['threats_detected']}{Colors.RESET}/{result['total_records']}")
                
                # Save results
                output_file = f"analysis_results_{Path(csv_file_path).stem}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved: {Colors.WHITE}{output_file}{Colors.RESET}")
                
                # Generate report
                if generate_report:
                    report_file = self._generate_report(result, Path(csv_file_path).stem, llm_model)
                    if report_file:
                        print(f"Report: {Colors.WHITE}{report_file}{Colors.RESET}")
            else:
                print(f"{Colors.RED}❌ {response.json()['detail']}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}❌ {e}{Colors.RESET}")
    
    def _generate_report(self, results, source_name, llm_model):
        """Generate professional markdown report using AI analysis"""
        try:
            timestamp = datetime.now()
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            
            report_filename = output_dir / f"threat_report_{source_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
            
            # Generate report using the built-in report generator
            report_content = self._create_report_content(results, source_name, timestamp)
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return str(report_filename)
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️  Report generation failed: {e}{Colors.RESET}")
            return None
    
    def _create_report_content(self, results, source_name, timestamp):
        """Create professional report content"""
        total = results['total_records']
        threats = results['threats_detected']
        avg_conf = results['average_confidence']
        threat_rate = (threats / total * 100) if total > 0 else 0
        
        # Determine status
        if threat_rate > 50:
            status = "🔴 CRITICAL"
        elif threat_rate > 20:
            status = "🟠 WARNING"
        else:
            status = "🟢 NORMAL"
        
        # Start building report
        report = []
        
        # Header
        report.append(f"# 🛡️ Network Threat Analysis Report")
        report.append(f"")
        report.append(f"**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Analyzer:** {self.username or 'Unknown'}")
        report.append(f"**Source:** {source_name}")
        report.append(f"**Analysis ID:** {timestamp.strftime('%Y%m%d_%H%M%S')}")
        report.append(f"")
        report.append(f"---")
        report.append(f"")
        
        # Executive Summary
        report.append(f"## 📊 Executive Summary")
        report.append(f"")
        report.append(f"{status} **{threats} threats detected** out of {total:,} network traffic records analyzed.")
        report.append(f"")
        report.append(f"- **Threat Detection Rate:** {threat_rate:.1f}%")
        report.append(f"- **Average Confidence:** {avg_conf:.1%}")
        report.append(f"- **Records Analyzed:** {total:,}")
        report.append(f"- **Clean Records:** {total - threats:,}")
        report.append(f"")
        
        if threats > 0:
            report.append(f"⚠️ **Action Required:** Immediate investigation recommended for high-confidence threats.")
        else:
            report.append(f"✅ **Status:** Network traffic appears normal.")
        report.append(f"")
        
        # Key Findings
        predictions = results['predictions']
        threat_types = {}
        high_conf_threats = []
        
        for pred in predictions:
            if pred['is_threat']:
                threat_type = pred['prediction']
                threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
                if pred['confidence'] > 0.9:
                    high_conf_threats.append(pred)
        
        if threat_types:
            report.append(f"## 🔍 Key Findings")
            report.append(f"")
            report.append(f"### Threat Distribution")
            report.append(f"")
            for threat_type, count in sorted(threat_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / threats) * 100
                report.append(f"- **{threat_type}:** {count} incidents ({percentage:.1f}%)")
            report.append(f"")
        
        if high_conf_threats:
            report.append(f"### ⚠️ High-Confidence Threats")
            report.append(f"")
            report.append(f"**{len(high_conf_threats)} threats** detected with >90% confidence:")
            report.append(f"")
            for threat in high_conf_threats[:10]:
                report.append(f"- Record #{threat['record_id']}: {threat['prediction']} ({threat['confidence']:.1%} confidence)")
            if len(high_conf_threats) > 10:
                report.append(f"- *...and {len(high_conf_threats) - 10} more*")
            report.append(f"")
        
        # Threat Breakdown by Confidence
        if threats > 0:
            threats_list = [p for p in predictions if p['is_threat']]
            critical = [t for t in threats_list if t['confidence'] > 0.9]
            high = [t for t in threats_list if 0.7 < t['confidence'] <= 0.9]
            medium = [t for t in threats_list if 0.5 < t['confidence'] <= 0.7]
            low = [t for t in threats_list if t['confidence'] <= 0.5]
            
            report.append(f"## 📈 Threat Breakdown by Confidence")
            report.append(f"")
            report.append(f"| Severity | Count | Percentage | Action |")
            report.append(f"|----------|-------|------------|--------|")
            report.append(f"| 🔴 Critical (>90%) | {len(critical)} | {len(critical)/threats*100:.1f}% | Immediate action required |")
            report.append(f"| 🟠 High (70-90%) | {len(high)} | {len(high)/threats*100:.1f}% | Investigate soon |")
            report.append(f"| 🟡 Medium (50-70%) | {len(medium)} | {len(medium)/threats*100:.1f}% | Monitor closely |")
            report.append(f"| ⚪ Low (<50%) | {len(low)} | {len(low)/threats*100:.1f}% | Track for patterns |")
            report.append(f"")
        
        # LLM Analysis (if available)
        if results.get('llm_analysis'):
            report.append(f"## 🤖 AI-Powered Threat Analysis")
            report.append(f"")
            report.append(f"Detailed analysis of {len(results['llm_analysis'])} high-priority threats:")
            report.append(f"")
            
            for i, analysis in enumerate(results['llm_analysis'], 1):
                report.append(f"### Threat #{i} (Record #{analysis['record_id']})")
                report.append(f"")
                report.append(analysis['analysis'])
                report.append(f"")
                report.append(f"---")
                report.append(f"")
        
        # Recommendations
        report.append(f"## 💡 Recommendations")
        report.append(f"")
        
        if threat_rate > 50:
            report.append(f"### 🔴 Immediate Actions Required")
            report.append(f"")
            report.append(f"1. **Isolate affected systems** - Block suspicious traffic sources")
            report.append(f"2. **Activate incident response team** - High threat volume detected")
            report.append(f"3. **Review firewall rules** - Update security policies")
            report.append(f"4. **Conduct security audit** - Investigate potential breach")
        elif threat_rate > 20:
            report.append(f"### 🟠 Recommended Actions")
            report.append(f"")
            report.append(f"1. **Investigate high-confidence threats** - Review detailed logs")
            report.append(f"2. **Update security rules** - Adjust firewall configurations")
            report.append(f"3. **Monitor continuously** - Increase surveillance on affected segments")
            report.append(f"4. **Review access controls** - Verify user permissions")
        elif threats > 0:
            report.append(f"### 🟢 Suggested Actions")
            report.append(f"")
            report.append(f"1. **Review flagged records** - Validate threat classifications")
            report.append(f"2. **Monitor trends** - Track for pattern changes")
            report.append(f"3. **Update threat intelligence** - Refine detection rules")
            report.append(f"4. **Schedule regular scans** - Maintain vigilance")
        else:
            report.append(f"### ✅ Maintenance Actions")
            report.append(f"")
            report.append(f"1. **Continue monitoring** - Maintain current security posture")
            report.append(f"2. **Regular updates** - Keep security systems current")
            report.append(f"3. **Staff training** - Reinforce security awareness")
            report.append(f"4. **Periodic assessments** - Schedule routine security reviews")
        report.append(f"")
        
        # Detailed Threat Log
        if threats > 0:
            threats_list = [p for p in predictions if p['is_threat']]
            threats_list.sort(key=lambda x: x['confidence'], reverse=True)
            
            report.append(f"## 📋 Detailed Threat Log")
            report.append(f"")
            report.append(f"Top {min(len(threats_list), 20)} threats (sorted by confidence):")
            report.append(f"")
            report.append(f"| Record ID | Threat Type | Confidence | Status |")
            report.append(f"|-----------|-------------|------------|--------|")
            
            for threat in threats_list[:20]:
                conf_str = f"{threat['confidence']:.1%}"
                status = "🔴" if threat['confidence'] > 0.9 else "🟠" if threat['confidence'] > 0.7 else "🟡"
                report.append(f"| {threat['record_id']} | {threat['prediction']} | {conf_str} | {status} |")
            
            if len(threats_list) > 20:
                report.append(f"")
                report.append(f"*... and {len(threats_list) - 20} more threats in full dataset*")
            report.append(f"")
        
        # Footer
        report.append(f"---")
        report.append(f"")
        report.append(f"## 📌 Report Metadata")
        report.append(f"")
        report.append(f"- **Total Records Analyzed:** {total:,}")
        report.append(f"- **Threats Detected:** {threats:,}")
        report.append(f"- **Average Confidence:** {avg_conf:.2%}")
        report.append(f"- **LLM Analysis:** {'Enabled' if results.get('llm_analysis') else 'Disabled'}")
        report.append(f"- **Report Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"- **Analyzer:** {self.username or 'Unknown User'}")
        report.append(f"")
        report.append(f"---")
        report.append(f"")
        report.append(f"*This report was automatically generated by the NIRVANA Network Threat Detector AI system.*")
        report.append(f"*For questions or concerns, contact your security team.*")
        
        return "\n".join(report)
    
    def list_models(self):
        """List models"""
        headers = self.get_headers()
        if not headers:
            return
        
        try:
            response = requests.get(f"{self.base_url}/models", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print(f"\n{Colors.CYAN}AVAILABLE MODELS{Colors.RESET}\n")
                for model in data['models']:
                    marker = f"{Colors.GREEN}✓{Colors.RESET}" if model == data['recommended'] else " "
                    print(f"{marker} {model}")
        except Exception as e:
            print(f"{Colors.RED}❌ {e}{Colors.RESET}")
    
    def health_check(self):
        """Health check"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"\n{Colors.CYAN}API HEALTH{Colors.RESET}\n")
                print(f"Status: {Colors.GREEN}{data['status']}{Colors.RESET}")
                print(f"Model: {Colors.GREEN if data['ml_model'] else Colors.RED}{'✓' if data['ml_model'] else '✗'}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}❌ {e}{Colors.RESET}")


def main():
    client = ThreatDetectorClient()
    
    if len(sys.argv) == 1:
        client.show_main_menu()
    else:
        command = sys.argv[1].lower()
        if command == "login":
            client.login(sys.argv[2] if len(sys.argv) > 2 else None,
                        sys.argv[3] if len(sys.argv) > 3 else None)
        elif command == "analyze":
            if len(sys.argv) > 2:
                client._print_banner()
                client.analyze_csv(sys.argv[2])


if __name__ == "__main__":
    main()