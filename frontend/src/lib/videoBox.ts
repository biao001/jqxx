import type { CSSProperties } from 'react';

/**
 * 把后端返回的视频像素坐标 bbox [x1,y1,x2,y2] 换算成叠加层的 CSS 绝对定位。
 * 处理 object-contain 模式下视频在容器内的居中留白(letterbox)。
 */
export function videoBoxStyle(video: HTMLVideoElement | null, bbox?: number[] | null): CSSProperties | null {
  if (!video || !bbox || bbox.length !== 4 || video.videoWidth === 0 || video.videoHeight === 0) return null;
  const rect = video.getBoundingClientRect();
  const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
  const renderedWidth = video.videoWidth * scale;
  const renderedHeight = video.videoHeight * scale;
  const offsetX = (rect.width - renderedWidth) / 2;
  const offsetY = (rect.height - renderedHeight) / 2;
  const [x1, y1, x2, y2] = bbox;

  return {
    left: `${offsetX + x1 * scale}px`,
    top: `${offsetY + y1 * scale}px`,
    width: `${Math.max(1, (x2 - x1) * scale)}px`,
    height: `${Math.max(1, (y2 - y1) * scale)}px`,
  };
}
