#ifndef VOXBLOX_ROS_CONVERSIONS_H_
#define VOXBLOX_ROS_CONVERSIONS_H_

#include <algorithm>
#include <memory>
#include <vector>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <std_msgs/ColorRGBA.h>

#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/mesh/mesh.h>
#include <voxblox/utils/color_maps.h>
#include <voxblox_msgs/Layer.h>

struct EIGEN_ALIGN16 SemanticPointColored {
  PCL_ADD_POINT4D;  // Adds x, y, z, and padding
  PCL_ADD_RGB;      // Adds color
  float semantic_state[kTotalNumberOf2DLabels];  // Evidential confidence vector
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Ensures alignment
};
POINT_CLOUD_REGISTER_POINT_STRUCT(
  SemanticPointColored,
  (float, x, x)(float, y, y)(float, z, z)(uint32_t, rgba, rgba)
  (float[kTotalNumberOf2DLabels], semantic_state, semantic_state)
)

struct EIGEN_ALIGN16 EvDSemanticPointColored {
  PCL_ADD_POINT4D;  // Adds x, y, z, and padding
  PCL_ADD_RGB;      // Adds color
  float semantic_state[kTotalNumberOf2DLabels];  // Evidential confidence vector
  float depth_uncertainty[2];  // Depth uncertainty
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Ensures alignment
};
POINT_CLOUD_REGISTER_POINT_STRUCT(
  EvDSemanticPointColored,
  (float, x, x)(float, y, y)(float, z, z)(uint32_t, rgba, rgba)
  (float[kTotalNumberOf2DLabels], semantic_state, semantic_state)
  (float[2], depth_uncertainty, depth_uncertainty)
)

struct EIGEN_ALIGN16 EvDPointColored {
  PCL_ADD_POINT4D;  // Adds x, y, z, and padding
  PCL_ADD_RGB;      // Adds color
  float depth_uncertainty[2];  // Depth uncertainty
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Ensures alignment
};
POINT_CLOUD_REGISTER_POINT_STRUCT(
  EvDPointColored,
  (float, x, x)(float, y, y)(float, z, z)(uint32_t, rgba, rgba)
  (float[2], depth_uncertainty, depth_uncertainty)
)

struct EIGEN_ALIGN16 SemanticPoint {
  PCL_ADD_POINT4D;  // Adds x, y, z, and padding
  float semantic_state[kTotalNumberOf2DLabels];  // Evidential confidence vector
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Ensures alignment
};
POINT_CLOUD_REGISTER_POINT_STRUCT(
  SemanticPoint,
  (float, x, x)(float, y, y)(float, z, z)
  (float[kTotalNumberOf2DLabels], semantic_state, semantic_state)
)

struct EIGEN_ALIGN16 EvDSemanticPoint {
  PCL_ADD_POINT4D;  // Adds x, y, z, and padding
  float semantic_state[kTotalNumberOf2DLabels];  // Evidential confidence vector
  float depth_uncertainty[2];  // Depth uncertainty
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Ensures alignment
};
POINT_CLOUD_REGISTER_POINT_STRUCT(
  EvDSemanticPoint,
  (float, x, x)(float, y, y)(float, z, z)
  (float[kTotalNumberOf2DLabels], semantic_state, semantic_state)
  (float[2], depth_uncertainty, depth_uncertainty)
)

struct EIGEN_ALIGN16 EvDPoint {
  PCL_ADD_POINT4D;  // Adds x, y, z, and padding
  float depth_uncertainty[2];  // Depth uncertainty
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Ensures alignment
};
POINT_CLOUD_REGISTER_POINT_STRUCT(
  EvDPoint,
  (float, x, x)(float, y, y)(float, z, z)
  (float[2], depth_uncertainty, depth_uncertainty)
)
namespace voxblox {

enum class MapDerializationAction : uint8_t {
  kUpdate = 0u,
  kMerge = 1u,
  kReset = 2u
};

inline void colorVoxbloxToMsg(const Color& color,
                              std_msgs::ColorRGBA* color_msg) {
  CHECK_NOTNULL(color_msg);
  color_msg->r = color.r / 255.0;
  color_msg->g = color.g / 255.0;
  color_msg->b = color.b / 255.0;
  color_msg->a = color.a / 255.0;
}

inline void colorMsgToVoxblox(const std_msgs::ColorRGBA& color_msg,
                              Color* color) {
  CHECK_NOTNULL(color);
  color->r = static_cast<uint8_t>(color_msg.r * 255.0);
  color->g = static_cast<uint8_t>(color_msg.g * 255.0);
  color->b = static_cast<uint8_t>(color_msg.b * 255.0);
  color->a = static_cast<uint8_t>(color_msg.a * 255.0);
}

inline void pointcloudToPclXYZRGB(
    const Pointcloud& ptcloud, const Colors& colors,
    pcl::PointCloud<pcl::PointXYZRGB>* ptcloud_pcl) {
  CHECK_NOTNULL(ptcloud_pcl);
  ptcloud_pcl->clear();
  ptcloud_pcl->reserve(ptcloud.size());
  for (size_t i = 0; i < ptcloud.size(); ++i) {
    pcl::PointXYZRGB point;
    point.x = ptcloud[i].x();
    point.y = ptcloud[i].y();
    point.z = ptcloud[i].z();

    point.r = colors[i].r;
    point.g = colors[i].g;
    point.b = colors[i].b;

    ptcloud_pcl->push_back(point);
  }
}

inline void pointcloudToPclXYZ(const Pointcloud& ptcloud,
                               pcl::PointCloud<pcl::PointXYZ>* ptcloud_pcl) {
  CHECK_NOTNULL(ptcloud_pcl);
  ptcloud_pcl->clear();
  ptcloud_pcl->reserve(ptcloud.size());
  for (size_t i = 0; i < ptcloud.size(); ++i) {
    pcl::PointXYZ point;
    point.x = ptcloud[i].x();
    point.y = ptcloud[i].y();
    point.z = ptcloud[i].z();

    ptcloud_pcl->push_back(point);
  }
}

inline void pointcloudToPclXYZI(const Pointcloud& ptcloud,
                                const std::vector<float>& intensities,
                                pcl::PointCloud<pcl::PointXYZI>* ptcloud_pcl) {
  CHECK_NOTNULL(ptcloud_pcl);
  CHECK_EQ(ptcloud.size(), intensities.size());
  ptcloud_pcl->clear();
  ptcloud_pcl->reserve(ptcloud.size());
  for (size_t i = 0; i < ptcloud.size(); ++i) {
    pcl::PointXYZI point;
    point.x = ptcloud[i].x();
    point.y = ptcloud[i].y();
    point.z = ptcloud[i].z();
    point.intensity = intensities[i];

    ptcloud_pcl->push_back(point);
  }
}

/// Check if all coordinates in the PCL point are finite.
template <typename PCLPoint>
inline bool isPointFinite(const PCLPoint& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

template <typename PCLPoint>
Color convertColor(const PCLPoint& point,
                   const std::shared_ptr<ColorMap>& color_map);

template <>
inline Color convertColor(const pcl::PointXYZRGB& point,
                          const std::shared_ptr<ColorMap>& /*color_map*/) {
  return Color(point.r, point.g, point.b, point.a);
}

template <>
inline Color convertColor(const pcl::PointXYZI& point,
                          const std::shared_ptr<ColorMap>& color_map) {
  return color_map->colorLookup(point.intensity);
}

template <>
inline Color convertColor(const pcl::PointXYZ& /*point*/,
                          const std::shared_ptr<ColorMap>& color_map) {
  return color_map->colorLookup(0);
}

/// Convert pointclouds of different PCL types to a voxblox pointcloud.
template <typename PCLPoint>
inline void convertPointcloud(
    const typename pcl::PointCloud<PCLPoint>& pointcloud_pcl,
    const std::shared_ptr<ColorMap>& color_map, Pointcloud* points_C,
    Colors* colors) {
  points_C->reserve(pointcloud_pcl.size());
  colors->reserve(pointcloud_pcl.size());
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));
    colors->emplace_back(
        convertColor<PCLPoint>(pointcloud_pcl.points[i], color_map));
  }
}
template <typename PCLColoredPoint>
inline Color convertColor(const PCLColoredPoint& point,
  const std::shared_ptr<ColorMap>& /*color_map*/) {
return Color(point.r, point.g, point.b, point.a);
}
template <typename PCLPointWithSemanticState>
inline SemanticState convertSemanticState(const PCLPointWithSemanticState& point){
  return Eigen::Map<const SemanticState>(point.semantic_state);
}
template <typename PCLPointWithDepthUncertainty>
inline DepthUncertainty convertDepthUncertainty(const PCLPointWithDepthUncertainty& point){
  return Eigen::Map<const DepthUncertainty>(point.depth_uncertainty);
}

template <typename PCLSemanticPoint>
inline void convertPointcloud(
    const typename pcl::PointCloud<PCLSemanticPoint>& pointcloud_pcl,
    Pointcloud* points_C,Semantics* semantics) {
  points_C->reserve(pointcloud_pcl.size());
  semantics->reserve(pointcloud_pcl.size());
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));
    semantics->emplace_back(Eigen::Map<const SemanticState>(pointcloud_pcl.points[i].semantic_state));
  }
}

template <typename PCLSemanticPointColored>
inline void convertPointcloud(
    const typename pcl::PointCloud<PCLSemanticPointColored>& pointcloud_pcl,
    const std::shared_ptr<ColorMap>& color_map, Pointcloud* points_C,
    Colors* colors, Semantics* semantics) {
  points_C->reserve(pointcloud_pcl.size());
  colors->reserve(pointcloud_pcl.size());
  semantics->reserve(pointcloud_pcl.size());
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));
    colors->emplace_back(
        convertColor<PCLSemanticPointColored>(pointcloud_pcl.points[i], color_map));
    
    semantics->emplace_back(Eigen::Map<const SemanticState>(pointcloud_pcl.points[i].semantic_state));
  }
}
template <typename PCLEvDSemanticPointColored>
inline void convertPointcloud(
    const typename pcl::PointCloud<PCLEvDSemanticPointColored>& pointcloud_pcl,
    const std::shared_ptr<ColorMap>& color_map, Pointcloud* points_C,
    Colors* colors, Semantics* semantics, DepthUncertainties* depth_uncertainties) {
  points_C->reserve(pointcloud_pcl.size());
  colors->reserve(pointcloud_pcl.size());
  semantics->reserve(pointcloud_pcl.size());
  depth_uncertainties->reserve(pointcloud_pcl.size());
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));
    colors->emplace_back(
        convertColor<PCLEvDSemanticPointColored>(pointcloud_pcl.points[i], color_map));

    SemanticState semantic_voxel_state = SemanticState::Ones();
    semantic_voxel_state.head(kTotalNumberOf2DLabels) = Eigen::Map<const Eigen::Matrix<FloatingPoint, kTotalNumberOf2DLabels, 1>>(pointcloud_pcl.points[i].semantic_state);
    //semantic_voxel_state(kFreeSpaceLabelId) ;
    semantics->push_back(semantic_voxel_state);
    depth_uncertainties->emplace_back(Eigen::Map<const DepthUncertainty>(pointcloud_pcl.points[i].depth_uncertainty));
  }
  // for (size_t j = 0; j < kTotalNumberOf2DLabels; ++j){
  //   ROS_INFO_STREAM("Semantic Label State: "<<pointcloud_pcl.points[0].semantic_state[j]);
  // }
  // ROS_INFO_STREAM_ONCE("Semantic Voxel State: "<<semantics->front().transpose());
  
}

template <typename PCLEvDSemanticPointColored>
inline void convertPointcloud(
    const typename pcl::PointCloud<PCLEvDSemanticPointColored>& pointcloud_pcl,
    Pointcloud* points_C, Semantics* semantics, DepthUncertainties* depth_uncertainties) {
  points_C->reserve(pointcloud_pcl.size());
  semantics->reserve(pointcloud_pcl.size());
  depth_uncertainties->reserve(pointcloud_pcl.size());
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));                              
    semantics->emplace_back(Eigen::Map<const SemanticState>(pointcloud_pcl.points[i].semantic_state));
    depth_uncertainties->emplace_back(Eigen::Map<const DepthUncertainty>(pointcloud_pcl.points[i].depth_uncertainty));
  }
}

template <typename PCLEvDPoint>
inline void convertPointcloud(
    const typename pcl::PointCloud<PCLEvDPoint>& pointcloud_pcl,
    Pointcloud* points_C,DepthUncertainties* depth_uncertainties) {
  points_C->reserve(pointcloud_pcl.size());
  depth_uncertainties->reserve(pointcloud_pcl.size());
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));
    depth_uncertainties->emplace_back(Eigen::Map<const DepthUncertainty>(pointcloud_pcl.points[i].depth_uncertainty));
  }
}
template <typename PCLEvDPointColored>
inline void convertPointcloud(
    const typename pcl::PointCloud<PCLEvDPointColored>& pointcloud_pcl,
    const std::shared_ptr<ColorMap>& color_map, Pointcloud* points_C,
    Colors* colors, DepthUncertainties* depth_uncertainties) {
  points_C->reserve(pointcloud_pcl.size());
  colors->reserve(pointcloud_pcl.size());
  depth_uncertainties->reserve(pointcloud_pcl.size());
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));
    colors->emplace_back(
            convertColor<PCLEvDPointColored>(pointcloud_pcl.points[i], color_map));
    depth_uncertainties->emplace_back(Eigen::Map<const DepthUncertainty>(pointcloud_pcl.points[i].depth_uncertainty));
  }
}

// Declarations
template <typename VoxelType>
void serializeLayerAsMsg(
    const Layer<VoxelType>& layer, const bool only_updated,
    voxblox_msgs::Layer* msg,
    const MapDerializationAction& action = MapDerializationAction::kUpdate);

/**
 * Returns true if could parse the data into the existing layer (all parameters
 * are compatible), false otherwise.
 * This function will use the deserialization action suggested by the layer
 * message.
 */
template <typename VoxelType>
bool deserializeMsgToLayer(const voxblox_msgs::Layer& msg,
                           Layer<VoxelType>* layer);

template <typename VoxelType>
bool deserializeMsgToLayer(const voxblox_msgs::Layer& msg,
                           const MapDerializationAction& action,
                           Layer<VoxelType>* layer);

}  // namespace voxblox

#endif  // VOXBLOX_ROS_CONVERSIONS_H_

#include "voxblox_ros/conversions_inl.h"
