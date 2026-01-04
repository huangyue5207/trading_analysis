# Nginx 部署指南

## 1. 安装 Nginx

在 Ubuntu 系统上执行以下命令：

```bash
# 更新软件包列表
sudo apt update

# 安装 nginx
sudo apt install nginx -y

# 启动 nginx 服务
sudo systemctl start nginx

# 设置 nginx 开机自启
sudo systemctl enable nginx

# 检查 nginx 状态
sudo systemctl status nginx
```

## 2. 部署文件

### 方法一：使用默认网站目录（推荐）

```bash
# 将项目文件复制到 nginx 默认网站目录
sudo cp index.html /var/www/html/
sudo cp BTCUSDT_15m_kline_mfi.png /var/www/html/
sudo cp ETHUSDT_15m_kline_mfi.png /var/www/html/

# 设置文件权限
sudo chown -R www-data:www-data /var/www/html/
sudo chmod -R 755 /var/www/html/
```

### 方法二：创建自定义网站目录

```bash
# 创建网站目录
sudo mkdir -p /var/www/trading_analysis

# 复制所有文件到网站目录
sudo cp index.html /var/www/trading_analysis/
sudo cp *.png /var/www/trading_analysis/

# 设置权限
sudo chown -R www-data:www-data /var/www/trading_analysis
sudo chmod -R 755 /var/www/trading_analysis
```

## 3. 配置 Nginx

### 方法一：使用默认配置（最简单）

如果使用默认目录 `/var/www/html/`，nginx 已经配置好了，直接访问即可。

### 方法二：创建自定义站点配置

```bash
# 创建站点配置文件
sudo nano /etc/nginx/sites-available/trading_analysis
```

在文件中添加以下内容：

```nginx
server {
    listen 80;
    server_name your_domain_or_ip;  # 替换为你的域名或IP地址，或使用 _ 表示所有域名

    root /var/www/trading_analysis;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # 优化图片加载
    location ~* \.(png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

保存文件后：

```bash
# 创建符号链接启用站点
sudo ln -s /etc/nginx/sites-available/trading_analysis /etc/nginx/sites-enabled/

# 测试 nginx 配置
sudo nginx -t

# 重新加载 nginx 配置
sudo systemctl reload nginx
```

## 4. 配置防火墙（如果需要）

```bash
# 允许 HTTP 流量
sudo ufw allow 'Nginx HTTP'

# 或者允许 HTTPS（如果配置了SSL）
sudo ufw allow 'Nginx HTTPS'

# 检查防火墙状态
sudo ufw status
```

## 5. 访问网站

- 如果使用默认配置：访问 `http://your_server_ip/`
- 如果使用自定义配置：访问 `http://your_server_ip/` 或 `http://your_domain/`

## 6. 常用命令

```bash
# 启动 nginx
sudo systemctl start nginx

# 停止 nginx
sudo systemctl stop nginx

# 重启 nginx
sudo systemctl restart nginx

# 重新加载配置（不中断服务）
sudo systemctl reload nginx

# 查看 nginx 错误日志
sudo tail -f /var/log/nginx/error.log

# 查看 nginx 访问日志
sudo tail -f /var/log/nginx/access.log
```

## 7. 故障排查

如果无法访问：

1. 检查 nginx 是否运行：`sudo systemctl status nginx`
2. 检查配置文件语法：`sudo nginx -t`
3. 检查防火墙设置：`sudo ufw status`
4. 查看错误日志：`sudo tail -f /var/log/nginx/error.log`
5. 检查文件权限：确保 `/var/www/html/` 或你的网站目录有正确的权限

