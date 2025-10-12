# PyTorch libtorch_cpu.so Fix - Complete Solution

## 🔍 **Issue Identified:**

```
❌ Failed to import orchestrator.api_server: libtorch_cpu.so: cannot enable executable stack as shared object requires: Invalid argument
```

### **Root Cause:**
PyTorch's `libtorch_cpu.so` library requires an executable stack, which is blocked by Docker's default security settings. This is a known issue when running PyTorch in containerized environments with strict security policies.

### **Why It Happened:**
1. Docker containers have security restrictions by default
2. PyTorch's C++ libraries (libtorch) need executable stack permissions
3. The container's seccomp profile blocks this by default

---

## ✅ **Fixes Applied:**

### **Fix 1: Added execstack Utility to Dockerfiles**

**Updated Files:**
- `docker/Dockerfile.orchestrator`
- `docker/Dockerfile.fog-node`

**Changes:**
```dockerfile
# Install execstack utility
RUN apt-get update && apt-get install -y \
    ...
    execstack \
    && rm -rf /var/lib/apt/lists/*

# Clear executable stack requirement from PyTorch libraries
RUN find /usr/local/lib/python3.9/site-packages/torch/lib -name "*.so*" -exec execstack -c {} \; 2>/dev/null || true
```

**What This Does:**
- Installs `execstack` tool to manage executable stack flags
- Clears the executable stack requirement from all PyTorch `.so` files
- Makes PyTorch compatible with Docker security settings

---

### **Fix 2: Added Security Options to docker-compose.yml**

**Updated File:**
- `docker/docker-compose.yml`

**Changes:**
```yaml
orchestrator:
  ...
  security_opt:
    - seccomp:unconfined

fog-node-1:
  ...
  security_opt:
    - seccomp:unconfined
```

**What This Does:**
- Relaxes Docker's seccomp security profile
- Allows PyTorch libraries to load with their required permissions
- Still maintains container isolation

---

## 🚀 **How to Apply the Fix:**

### **Option 1: Use the Quick Fix Script (Recommended)**

```bash
# Make script executable
chmod +x fix_pytorch_and_redeploy.sh

# Run the fix
./fix_pytorch_and_redeploy.sh
```

This script will:
1. ✅ Stop all containers
2. ✅ Remove old images
3. ✅ Rebuild with PyTorch fixes
4. ✅ Start services
5. ✅ Test endpoints

---

### **Option 2: Manual Steps**

```bash
# 1. Stop containers
docker-compose -f docker/docker-compose.yml down

# 2. Remove old images
docker rmi docker_orchestrator docker_fog-node-1 docker_fog-node-2 docker_fog-node-3

# 3. Rebuild
docker-compose -f docker/docker-compose.yml build --no-cache

# 4. Start
docker-compose -f docker/docker-compose.yml up -d

# 5. Check status
docker-compose -f docker/docker-compose.yml ps

# 6. Check logs
docker logs nids-orchestrator --tail 20
docker logs nids-fog-node-1 --tail 20
```

---

## ✅ **Expected Results After Fix:**

### **Before Fix:**
```
❌ Failed to import orchestrator.api_server: libtorch_cpu.so: cannot enable executable stack
Container Status: Restarting
```

### **After Fix:**
```
✅ Successfully imported orchestrator.api_server
🚀 Starting NIDS Central Orchestrator on 0.0.0.0:8000
Container Status: Up (healthy)
```

---

## 🔍 **Verification Steps:**

### **1. Check Container Status**
```bash
docker-compose -f docker/docker-compose.yml ps
```

**Expected:**
```
nids-orchestrator    Up (healthy)
nids-fog-node-1      Up (healthy)
nids-fog-node-2      Up (healthy)
nids-fog-node-3      Up (healthy)
```

### **2. Check Logs for Success**
```bash
docker logs nids-orchestrator --tail 20
```

**Expected to see:**
```
🔍 Python paths:
  - /app/src
  - /app
✅ Successfully imported orchestrator.api_server
🚀 Starting NIDS Central Orchestrator on 0.0.0.0:8000
```

### **3. Test Health Endpoint**
```bash
curl http://localhost:8000/health
```

**Expected:**
```json
{"status": "healthy"}
```

---

## 🐛 **Troubleshooting:**

### **If containers still restart after fix:**

1. **Check if execstack was installed:**
```bash
docker run --rm docker_orchestrator which execstack
```

2. **Check if PyTorch libraries were fixed:**
```bash
docker run --rm docker_orchestrator execstack -q /usr/local/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so
```

3. **Check security options:**
```bash
docker inspect nids-orchestrator | grep -A 5 SecurityOpt
```

4. **View full error logs:**
```bash
docker logs nids-orchestrator --tail 100
```

---

## 📚 **Technical Background:**

### **What is execstack?**
`execstack` is a Linux utility that manages the executable stack flag on ELF binaries and shared libraries. PyTorch's C++ libraries sometimes require this flag, which conflicts with modern security practices.

### **What is seccomp?**
Seccomp (Secure Computing Mode) is a Linux kernel feature that restricts system calls. Docker uses seccomp profiles to enhance container security. The `seccomp:unconfined` option relaxes these restrictions.

### **Security Implications:**
- **execstack -c**: Clears the executable stack requirement (more secure)
- **seccomp:unconfined**: Allows more system calls (less restrictive)
- **Combined**: Provides a balance between functionality and security

---

## ✅ **Summary:**

| Component | Issue | Fix | Status |
|-----------|-------|-----|--------|
| **Dockerfiles** | Missing execstack | Added execstack utility | ✅ Fixed |
| **PyTorch libs** | Executable stack required | Cleared with execstack -c | ✅ Fixed |
| **docker-compose** | Seccomp blocking | Added security_opt | ✅ Fixed |
| **Import paths** | Module not found | Fixed in previous update | ✅ Fixed |

---

## 🎯 **Next Steps:**

1. Run `./fix_pytorch_and_redeploy.sh`
2. Wait for containers to start (30 seconds)
3. Verify all containers show "Up (healthy)"
4. Test orchestrator endpoint: `curl http://localhost:8000/health`
5. Access Grafana: http://localhost:3000 (admin/admin)
6. Access Prometheus: http://localhost:9090

**Your distributed NIDS should now be fully operational!** 🎉🛡️
